/*
 * CWLB-SpMM standalone production CLI with candidate-guided S selection.
 * This source copy keeps the existing CSR-only loader, degree reordering,
 * cost-weighted partitioner and GPU kernels. It does not modify root sources.
 * --candidates evaluates the three NNZ-dependent w candidates from Section 4.2
 * using S = max(1, ceil(NNZ / (device_compute_units * w))). Degree reordering
 * and deterministic features are shared; each S rebuilds task descriptors and
 * runs the unchanged warm-up/timing loop from include/config.h.
 * --candidate-search [count=6] or --candidate-search --step STEP then prints
 * an inclusive interval and its integer grid, asks yes/no, and evaluates the
 * grid only after confirmation. The final minimum includes initial candidates.
 * It appends S/time blocks to ../resources/results.txt only after successful
 * confirmed execution. Cancellation/EOF does not create or change that file.
 * All existing wave, --S, and explicit range-sweep commands remain available.
 *
 * Build on NVIDIA after initializing the configured oneAPI environment:
 *   cmake -S test/CWLBSource -B test/CWLBSource/build
 *   cmake --build test/CWLBSource/build -j
 * From a working directory resolving include/config.h's data/resource paths:
 *   ./cwlb-spmm cora 128 --candidates
 *   ./cwlb-spmm cora 128 --candidate-search
 *   ./cwlb-spmm cora 128 --candidate-search 10
 *   ./cwlb-spmm cora 128 --candidate-search --step 3
 * Calibrate with costWrite first, or supply an existing compatible cost file.
 * Candidate tables contain candidate ID, w, S and mean SpMM time in us; stdout
 * retains individual event times. Means exclude warm-ups, preprocessing, data
 * transfer and file I/O. No end-to-end timing or global-optimum claim is made.
 * Equal times retain the first evaluated candidate. Integer grids are bounded
 * to 10000 points; narrow intervals may contain fewer than the requested count.
 * This simplified CLI has no external reference-file dependency or automatic
 * correctness verdict. Independent CPU/GPU and CLI validation is supplied by
 * ../cwlbSourceCandidateTest; its retained data belong under resources/.
 */
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>

#include <sycl/sycl.hpp>

#include "Matrix.h"
#include "config.h"
#include "data.h"
#include "parallelMatrix.h"
#include "utils.h"

enum class HyperParamType {
    WaveW,
    DirectS,
    SRange,
    Candidates,
    CandidateSearch
};

struct SingleSResult {
    int s = 0;
    double capacity = 0.0;
    std::size_t task_count = 0;
    double average_us = 0.0;
    std::vector<double> timed_samples_us;
    Defined::RowMajorMatrixXf output;
};

struct CandidateResult {
    int number = 0;
    int wave = 0;
    int s = 0;
    double average_us = 0.0;
};

// NNZ-based wave candidates, listed in evaluation priority order.
std::array<int, 3> recommended_candidate_w(std::uint64_t nnz) {
    if (nnz < 100000ULL) return {64, 32, 16};
    if (nnz < 500000ULL) return {128, 64, 32};
    if (nnz < 1000000ULL) return {256, 128, 64};
    if (nnz < 5000000ULL) return {512, 256, 128};
    if (nnz < 10000000ULL) return {1024, 512, 256};
    if (nnz < 50000000ULL) return {4096, 2048, 1024};
    return {8192, 4096, 2048};
}

constexpr int MAX_CANDIDATE_SEARCH_POINTS = 10000;

int positive_search_argument(const std::string& text) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (consumed != text.size() || value <= 0) {
        throw std::invalid_argument("Search count/step must be a positive integer.");
    }
    return value;
}

// Extend toward the best endpoint, or bracket the best middle candidate.
std::pair<int, int> candidate_search_bounds(
    const std::array<int, 3>& values, std::size_t best_index)
{
    if (best_index >= values.size() || values[0] <= 0 ||
        values[1] < values[0] || values[2] < values[1]) {
        throw std::invalid_argument("Invalid ordered candidate values or best index.");
    }
    std::int64_t lower = values[0], upper = values[2];
    if (best_index == 0) {
        lower = 2LL * values[0] - values[1];
        upper = values[1];
    } else if (best_index == 2) {
        lower = values[1];
        upper = 2LL * values[2] - values[1];
    }
    lower = std::max<std::int64_t>(1, lower);
    upper = std::min<std::int64_t>(std::numeric_limits<int>::max(), upper);
    return {static_cast<int>(lower), static_cast<int>(upper)};
}

// Build a near-uniform integer grid with both endpoints when count > 1.
std::vector<int> candidate_search_by_count(int lower, int upper, int requested) {
    if (lower <= 0 || upper < lower || requested <= 0) {
        throw std::invalid_argument("Invalid search interval or point count.");
    }
    const std::int64_t width = static_cast<std::int64_t>(upper) - lower;
    const int count = static_cast<int>(std::min<std::int64_t>(requested, width + 1));
    if (count > MAX_CANDIDATE_SEARCH_POINTS) {
        throw std::invalid_argument("Search plan exceeds 10000 points; reduce the count.");
    }
    if (count == 1) return {static_cast<int>(lower + width / 2)};
    std::vector<int> values;
    values.reserve(count);
    for (int index = 0; index < count; ++index) {
        const std::int64_t offset =
            (width * index + (count - 1) / 2) / (count - 1);
        values.push_back(static_cast<int>(lower + offset));
    }
    return values;
}

// Keep the upper endpoint even when the requested step does not land on it.
std::vector<int> candidate_search_by_step(int lower, int upper, int step) {
    if (lower <= 0 || upper < lower || step <= 0) {
        throw std::invalid_argument("Invalid search interval or step.");
    }
    const std::int64_t width = static_cast<std::int64_t>(upper) - lower;
    const std::int64_t count = width / step + 1 + (width % step != 0);
    if (count > MAX_CANDIDATE_SEARCH_POINTS) {
        throw std::invalid_argument("Search plan exceeds 10000 points; increase the step.");
    }
    std::vector<int> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::int64_t value = lower; value <= upper; value += step) {
        values.push_back(static_cast<int>(value));
    }
    if (values.back() != upper) values.push_back(upper);
    return values;
}

// Require explicit confirmation before running the neighborhood search.
bool confirm_candidate_search() {
    std::string answer;
    for (;;) {
        std::cout << "Continue CWLB-SpMM for these S values? (yes/no): " << std::flush;
        if (!std::getline(std::cin, answer)) return false;
        const auto first = answer.find_first_not_of(" \t\r\n");
        const auto last = answer.find_last_not_of(" \t\r\n");
        answer = first == std::string::npos ? "" : answer.substr(first, last - first + 1);
        std::transform(answer.begin(), answer.end(), answer.begin(),
                       [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        if (answer == "yes") return true;
        if (answer == "no") return false;
        std::cout << "Please enter yes or no.\n";
    }
}

constexpr const char* S_SEARCH_RESULT_PATH = "../resources/results.txt";

/*
 * CWLB-SpMM production degree reordering on the host CPU.
 *
 * The oneDPL par_unseq policy parallelizes degree generation, row sorting,
 * prefix-sum construction, and the disjoint CSR row copies. Rows are ordered
 * by (degree, original_row_id), so equal-degree ties are deterministic. The
 * resulting Origin_local permutation preserves the mapping needed by the
 * kernel to restore output rows to their original semantic order.
 */
CSR_Order sortCSRByDegreeDPLCPU(const CSR& csr) {
    CSR_Order result;
    if (csr.RowOffset.size() < 2) {
        return result;
    }
    if (csr.ColInd.size() != csr.Value.size()) {
        throw std::runtime_error("CSR ColInd and Value size mismatch.");
    }
    const std::size_t row_count = csr.RowOffset.size() - 1;
    const std::size_t nnz_count = csr.ColInd.size();
    if (row_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        nnz_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "DPL CPU degree reordering requires row and NNZ counts to fit in int.");
    }
    const int num_rows = static_cast<int>(row_count);
    const int nnz = static_cast<int>(nnz_count);
    const auto first = oneapi::dpl::counting_iterator<int>(0);
    const auto last = first + num_rows;
    const auto policy = oneapi::dpl::execution::par_unseq;
    /*---------------- Degree generation and row ordering ----------------*/
    std::vector<int> degrees(num_rows);
    std::transform(
        policy,
        first,
        last,
        degrees.begin(),
        [&csr](int row) {
            return csr.RowOffset[row + 1] - csr.RowOffset[row];
        });
    result.Origin_local.resize(num_rows);
    std::copy(policy, first, last, result.Origin_local.begin());
    std::sort(
        policy,
        result.Origin_local.begin(),
        result.Origin_local.end(),
        [&degrees](int lhs, int rhs) {
            if (degrees[lhs] != degrees[rhs]) {
                return degrees[lhs] < degrees[rhs];
            }
            return lhs < rhs;
        });
    /*---------------- Reordered CSR offsets ----------------*/
    std::vector<int> sorted_degrees(num_rows);
    std::transform(
        policy,
        result.Origin_local.begin(),
        result.Origin_local.end(),
        sorted_degrees.begin(),
        [&degrees](int row) { return degrees[row]; });
    result.RowOffset.resize(row_count + 1);
    result.RowOffset[0] = 0;
    oneapi::dpl::inclusive_scan(
        policy,
        sorted_degrees.begin(),
        sorted_degrees.end(),
        result.RowOffset.begin() + 1);
    /*---------------- Reordered CSR materialization ----------------*/
    result.ColInd.resize(nnz_count);
    result.Value.resize(nnz_count);
    std::for_each(
        policy,
        first,
        last,
        [&csr, &result](int sorted_row) {
            const int original_row = result.Origin_local[sorted_row];
            const int old_start = csr.RowOffset[original_row];
            const int old_end = csr.RowOffset[original_row + 1];
            const int new_start = result.RowOffset[sorted_row];
            for (int offset = 0; offset < old_end - old_start; ++offset) {
                result.ColInd[new_start + offset] =
                    csr.ColInd[old_start + offset];
                result.Value[new_start + offset] =
                    csr.Value[old_start + offset];
            }
        });
    if (result.RowOffset.back() != nnz) {
        throw std::runtime_error(
            "DPL CPU degree reordering produced an inconsistent NNZ count.");
    }
    return result;
}

// A full segment costs S load/compute pairs plus one atomic write-back.
double capacity_from_s(int s, const SpmmCostParams& costs) {
    const double edge_cost = costs.C_compute_ns + costs.C_load_ns;
    const int clamped_s = std::max(1, s);

    if (edge_cost <= 0.0) {
        return std::max(0.0, costs.C_atomic_ns);
    }

    return std::max(
        0.0,
        costs.C_atomic_ns + edge_cost * static_cast<double>(clamped_s));
}

// S = ceil(NNZ / (compute_units * wave)), clamped to a positive int.
int wave_based_s(
    const CSR_Order& sorted_csr,
    int compute_units,
    int wave)
{
    if (compute_units <= 0 || wave <= 0) {
        throw std::invalid_argument("compute_units and wave must be positive.");
    }

    if (sorted_csr.RowOffset.empty()) {
        return 1;
    }

    const std::uint64_t nnz =
        static_cast<std::uint64_t>(sorted_csr.RowOffset.back());
    const std::uint64_t denominator =
        static_cast<std::uint64_t>(compute_units) *
        static_cast<std::uint64_t>(wave);

    std::uint64_t s = (nnz + denominator - 1) / denominator;
    s = std::max<std::uint64_t>(1, s);
    s = std::min<std::uint64_t>(
        s,
        static_cast<std::uint64_t>(std::numeric_limits<int>::max()));

    return static_cast<int>(s);
}

// Reuse the same seeded feature matrix across the tested S values.
Defined::RowMajorMatrixXf deterministic_features(int rows, int cols) {
    Defined::RowMajorMatrixXf features(rows, cols);
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            features(row, col) = distribution(generator);
        }
    }

    return features;
}

SingleSResult run_single_s(
    sycl::queue& queue,
    const CSR_Order& sorted_csr,
    const Defined::RowMajorMatrixXf& features,
    int feature_dim,
    const SpmmCostParams& costs,
    int s,
    bool retain_timed_samples = false)
{
    SingleSResult result;
    result.s = s;
    result.capacity = capacity_from_s(s, costs);

    // Rebuild task descriptors for this S; reordered CSR and features stay fixed.
    const std::vector<ParallelDataWeighted> tasks =
        costWeightedTaskPartitioning(sorted_csr, costs, result.capacity);

    result.task_count = tasks.size();

    double elapsed_us = 0.0;
    double accumulated_us = 0.0;
    const std::size_t device_max_work_group_size =
        queue.get_device().get_info<
            sycl::info::device::max_work_group_size>();
    // Use feature striding only when K exceeds the legal work-group size.
    const bool use_large_k_fallback =
        static_cast<std::size_t>(feature_dim) >
        device_max_work_group_size;

    const std::string device_name =
        queue.get_device().get_info<sycl::info::device::name>();

    /*---------------- Kernel warm-up and event timing ----------------*/
    // LOOPCOUNT includes warm-up; only the remaining launches enter the mean.
    for (int iteration = 0; iteration < LOOPCOUNT; ++iteration) {
        if (use_large_k_fallback) {
            result.output = spmmKernelLargeKFallback(
                queue,
                sorted_csr,
                features,
                feature_dim,
                feature_dim,
                tasks,
                elapsed_us);
        } else {
            result.output = spmmKernel(
                queue,
                sorted_csr,
                features,
                feature_dim,
                feature_dim,
                tasks,
                elapsed_us);
        }

        const bool warmup = iteration < WARMUPCOUNT;
        std::cout << std::fixed << std::setprecision(6)
                  << (warmup ? "[Warm-up " : "[Iteration ")
                  << (warmup ? iteration + 1 : iteration - WARMUPCOUNT + 1)
                  << '/' << (warmup ? WARMUPCOUNT : LOOPCOUNT - WARMUPCOUNT)
                  << "] device=" << device_name
                  << " K=" << feature_dim
                  << " time=" << elapsed_us << " us\n";

        if (!warmup) {
            accumulated_us += elapsed_us;
            if (retain_timed_samples) result.timed_samples_us.push_back(elapsed_us);
        }
    }

    result.average_us =
        accumulated_us / static_cast<double>(LOOPCOUNT - WARMUPCOUNT);

    return result;
}

void print_single_s_result(
    const std::string& dataset,
    int vertex_count,
    int edge_count,
    const SpmmCostParams& costs,
    const SingleSResult& result)
{
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Dataset: " << dataset << '\n';
    std::cout << "V: " << vertex_count << ", E: " << edge_count << '\n';
    std::cout << "SpMM time: " << result.average_us << " us\n";
    std::cout << "costs: C_compute_ns=" << costs.C_compute_ns
              << " C_load_ns=" << costs.C_load_ns
              << " C_store_ns=" << costs.C_store_ns
              << " C_atomic_ns=" << costs.C_atomic_ns << '\n';
}

int main(int argc, char* argv[]) {
    if (argc != 4 && argc != 5 && argc != 6) {
        std::cerr
            << "Usage:\n"
            << "  ./cwlb-spmm [dataset] [K] [wave]\n"
            << "  ./cwlb-spmm [dataset] [K] --S [S]\n"
            << "  ./cwlb-spmm [dataset] [K] --candidates\n"
            << "  ./cwlb-spmm [dataset] [K] --candidate-search [count=6]\n"
            << "  ./cwlb-spmm [dataset] [K] --candidate-search --step [step]\n"
            << "  ./cwlb-spmm [dataset] [K] "
            << "[S_begin] [S_max] [S_step]\n";
        return 1;
    }

    try {
        const std::string dataset = argv[1];
        const int feature_dim = std::stoi(argv[2]);

        /*---------------- Command-line parameters ----------------*/
        HyperParamType parameter_type = HyperParamType::WaveW;
        int parameter_value = 0;
        int s_max = 0;
        int s_step = 0;
        int search_count = 6;
        int search_step = 0;

        if (std::string(argv[3]) == "--candidate-search") {
            parameter_type = HyperParamType::CandidateSearch;
            if (argc == 5) {
                search_count = positive_search_argument(argv[4]);
            } else if (argc == 6 && std::string(argv[4]) == "--step") {
                search_step = positive_search_argument(argv[5]);
            } else if (argc != 4) {
                throw std::invalid_argument(
                    "Use --candidate-search [count] or --candidate-search --step step.");
            }
        } else if (argc == 4) {
            if (std::string(argv[3]) == "--candidates") {
                parameter_type = HyperParamType::Candidates;
            } else {
                parameter_value = std::stoi(argv[3]);
            }
        } else if (argc == 5 && std::string(argv[3]) == "--S") {
            parameter_type = HyperParamType::DirectS;
            parameter_value = std::stoi(argv[4]);
        } else if (argc == 6) {
            parameter_type = HyperParamType::SRange;
            parameter_value = std::stoi(argv[3]);
            s_max = std::stoi(argv[4]);
            s_step = std::stoi(argv[5]);
        } else {
            throw std::invalid_argument("The five-argument form must use --S.");
        }

        const bool candidate_mode =
            parameter_type == HyperParamType::Candidates ||
            parameter_type == HyperParamType::CandidateSearch;
        if (feature_dim <= 0 || (!candidate_mode && parameter_value <= 0)) {
            throw std::invalid_argument("K and the hyperparameter must be positive.");
        }

        if (candidate_mode && (WARMUPCOUNT < 0 || LOOPCOUNT <= WARMUPCOUNT)) {
            throw std::invalid_argument(
                "Candidate timing requires 0 <= WARMUPCOUNT < LOOPCOUNT.");
        }

        if (parameter_type == HyperParamType::SRange &&
            (s_max <= 0 || s_step <= 0)) {
            throw std::invalid_argument(
                "S_begin, S_max and S_step must be positive.");
        }

        if (parameter_type == HyperParamType::SRange &&
            parameter_value > s_max) {
            throw std::invalid_argument("S_begin must not be greater than S_max.");
        }

        /*---------------- Graph input and device setup ----------------*/
        int vertex_count = 0;
        int edge_count = 0;
        const CSR adjacency =
            loadDatasetToCSR(dataset, vertex_count, edge_count);

        sycl::queue queue(
            sycl::gpu_selector_v,
            sycl::property::queue::enable_profiling{});

        // Calibration is a separate step; load the weights saved by costWrite.
        SpmmCostParams costs;
        if (!read_costs_one_per_line(costs)) {
            throw std::runtime_error(
                "Cannot read microbenchmark cost file: " + COSTFILEPATH +
                ". Run ./costWrite first.");
        }

        /*---------------- Shared input preprocessing ----------------*/
        const auto features =
            deterministic_features(vertex_count, feature_dim);

        const CSR_Order sorted_csr =
            sortCSRByDegreeDPLCPU(adjacency);

        /*---------------- Candidate evaluation ----------------*/
        if (candidate_mode) {
            const int compute_units = queue.get_device().get_info<
                sycl::info::device::max_compute_units>();
            const std::uint64_t nnz = sorted_csr.RowOffset.empty()
                ? 0ULL : static_cast<std::uint64_t>(sorted_csr.RowOffset.back());
            const auto waves = recommended_candidate_w(nnz);
            std::vector<CandidateResult> candidates;
            std::size_t best_index = 0;
            std::cout << "[Candidates] dataset=" << dataset
                      << " K=" << feature_dim << " NNZ=" << nnz
                      << " compute_units=" << compute_units
                      << " warmup_launches=" << WARMUPCOUNT
                      << " timed_launches=" << LOOPCOUNT - WARMUPCOUNT << '\n';

            for (std::size_t index = 0; index < waves.size(); ++index) {
                const int s = wave_based_s(sorted_csr, compute_units, waves[index]);
                std::cout << "[Candidates] running S" << index + 1
                          << " w=" << waves[index] << " S=" << s << '\n';
                const auto result = run_single_s(
                    queue, sorted_csr, features, feature_dim, costs, s, true);
                if (!std::isfinite(result.average_us) || result.average_us <= 0.0) {
                    throw std::runtime_error("Invalid candidate kernel time.");
                }
                print_single_s_result(dataset, vertex_count, edge_count, costs, result);
                std::cout << "candidate_samples_us\tS" << index + 1;
                for (double sample : result.timed_samples_us) std::cout << '\t' << sample;
                std::cout << '\n';
                candidates.push_back(CandidateResult{
                    static_cast<int>(index + 1), waves[index], s, result.average_us});
                if (result.average_us < candidates[best_index].average_us) best_index = index;
            }

            std::cout << "\n=== Candidate summary (SpMM kernel time, us) ===\n"
                      << "candidate\tw\tS\tSpMM_time_us\n";
            for (const auto& candidate : candidates) {
                std::cout << 'S' << candidate.number << '\t' << candidate.wave
                          << '\t' << candidate.s << '\t' << candidate.average_us << '\n';
            }
            const auto& best = candidates[best_index];
            std::cout << "Best among the three measured candidates: S" << best.number
                      << " (w=" << best.wave << ", S=" << best.s
                      << "), SpMM time=" << best.average_us << " us\n"
                      << "best_candidate=S" << best.number << " best_S=" << best.s
                      << " best_kernel_time_us=" << best.average_us << '\n';

            /*---------------- Confirmed neighborhood search ----------------*/
            if (parameter_type == HyperParamType::CandidateSearch) {
                const std::array<int, 3> initial_s = {
                    candidates[0].s, candidates[1].s, candidates[2].s};
                const auto bounds = candidate_search_bounds(initial_s, best_index);
                const auto search_values = search_step > 0
                    ? candidate_search_by_step(bounds.first, bounds.second, search_step)
                    : candidate_search_by_count(bounds.first, bounds.second, search_count);
                std::cout << "[Search plan] lower=" << bounds.first << " upper=" << bounds.second;
                if (search_step > 0) std::cout << " manual_step=" << search_step;
                else std::cout << " requested_count=" << search_count;
                std::cout << " actual_count=" << search_values.size() << '\n'
                          << "[Search plan] S values:";
                for (int s : search_values) std::cout << ' ' << s;
                std::cout << '\n';
                if (!confirm_candidate_search()) {
                    std::cout << "Search cancelled; results.txt unchanged.\n";
                    return 0;
                }

                // Include the initial winner when selecting the final best S.
                CandidateResult best_overall = best;
                std::string best_source = "initial_candidate_S" + std::to_string(best.number);
                std::vector<CandidateResult> search_results;
                for (std::size_t index = 0; index < search_values.size(); ++index) {
                    const int s = search_values[index];
                    std::cout << "[Search] running P" << index + 1 << " S=" << s << '\n';
                    const auto result = run_single_s(
                        queue, sorted_csr, features, feature_dim, costs, s, true);
                    if (!std::isfinite(result.average_us) || result.average_us <= 0.0) {
                        throw std::runtime_error("Invalid search kernel time.");
                    }
                    print_single_s_result(dataset, vertex_count, edge_count, costs, result);
                    std::cout << "search_samples_us\tP" << index + 1;
                    for (double sample : result.timed_samples_us) std::cout << '\t' << sample;
                    std::cout << '\n';
                    search_results.push_back(CandidateResult{
                        static_cast<int>(index + 1), 0, s, result.average_us});
                    if (result.average_us < best_overall.average_us) {
                        best_overall = search_results.back();
                        best_source = "search_P" + std::to_string(index + 1);
                    }
                }

                // Delay append until confirmation and completion; retain both phases.
                std::ostringstream saved;
                saved << std::fixed << std::setprecision(6)
                      << "cwlb-spmm-candidates\t" << dataset << '\t' << feature_dim << '\n';
                for (const auto& candidate : candidates) {
                    saved << candidate.s << '\t' << candidate.average_us << '\n';
                }
                saved << "cwlb-spmm-candidate-search\t" << dataset << '\t' << feature_dim << '\n';
                for (const auto& result : search_results) {
                    saved << result.s << '\t' << result.average_us << '\n';
                }
                std::ofstream output(S_SEARCH_RESULT_PATH, std::ios::out | std::ios::app);
                if (!output) {
                    throw std::runtime_error("Cannot append search results to " +
                                             std::string(S_SEARCH_RESULT_PATH));
                }
                output << saved.str();
                output.flush();
                if (!output) throw std::runtime_error("Failed to save candidate-search results.");

                std::cout << "\n=== Search summary (SpMM kernel time, us) ===\n"
                          << "point\tS\tSpMM_time_us\n";
                for (const auto& result : search_results) {
                    std::cout << 'P' << result.number << '\t' << result.s << '\t'
                              << result.average_us << '\n';
                }
                std::cout << "Best among all evaluated initial/search configurations: S="
                          << best_overall.s << ", SpMM time=" << best_overall.average_us << " us\n"
                          << "final_best_S=" << best_overall.s
                          << " final_best_kernel_time_us=" << best_overall.average_us
                          << " source=" << best_source << '\n'
                          << "Results appended to " << S_SEARCH_RESULT_PATH << '\n';
            }
            return 0;
        } else if (parameter_type == HyperParamType::SRange) {
            /*---------------- Explicit S sweep ----------------*/

            std::ofstream cost_file(
                S_SEARCH_RESULT_PATH,
                std::ios::out | std::ios::app);

            if (!cost_file.is_open()) {
                throw std::runtime_error(
                    "Cannot open S search result file: " +
                    std::string(S_SEARCH_RESULT_PATH));
            }

            cost_file << "cwlb-spmm\t" << dataset << '\t'
                      << feature_dim << '\n';
            cost_file << std::fixed << std::setprecision(6);
            cost_file.flush();

            int best_s = 0;
            double best_time_us = std::numeric_limits<double>::infinity();

            for (std::int64_t current_s = parameter_value;
                 current_s <= s_max;
                 current_s += s_step) {
                SingleSResult result = run_single_s(
                    queue,
                    sorted_csr,
                    features,
                    feature_dim,
                    costs,
                    static_cast<int>(current_s));

                print_single_s_result(
                    dataset, vertex_count, edge_count, costs, result);

                cost_file << result.s << '\t'
                          << result.average_us << '\n';
                cost_file.flush();

                if (!cost_file) {
                    throw std::runtime_error(
                        "Failed to append S search result file: " +
                        std::string(S_SEARCH_RESULT_PATH));
                }

                if (result.average_us < best_time_us) {
                    best_s = result.s;
                    best_time_us = result.average_us;
                }

            }

            std::cout << std::fixed << std::setprecision(6)
                      << "best_S=" << best_s
                      << " best_kernel_time_us=" << best_time_us << '\n';
        } else {

            /*---------------- Single-configuration execution ----------------*/
            int s = parameter_value;

            if (parameter_type == HyperParamType::WaveW) {
                const int compute_units = queue.get_device().get_info<
                    sycl::info::device::max_compute_units>();
                s = wave_based_s(sorted_csr, compute_units, parameter_value);
            }

            SingleSResult result = run_single_s(
                queue,
                sorted_csr,
                features,
                feature_dim,
                costs,
                s);

            print_single_s_result(
                dataset, vertex_count, edge_count, costs, result);
        }

    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
