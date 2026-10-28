// Operation calibration, weighted task construction, and validation helpers.
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>


namespace {
    double event_elapsed_ms_or_fallback(const sycl::event& e, bool profiling_enabled,
                                            std::chrono::steady_clock::time_point t0 = {},
                                            std::chrono::steady_clock::time_point t1 = {})
    {
        if (profiling_enabled) {
            uint64_t t_begin = 0, t_end = 0;
            try {
                t_begin = e.get_profiling_info<sycl::info::event_profiling::command_start>();
                t_end   = e.get_profiling_info<sycl::info::event_profiling::command_end>();
            } catch (...) {
                // If event profiling fails, use the caller-provided host timestamps.
                profiling_enabled = false;
            }
            if (profiling_enabled) {
                return double(t_end - t_begin) * 1e-6; // ns->ms
            }
        }
        // Host timing requires valid t0 and t1 supplied by the caller.
        auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return ms;
    }
} // namespace
// Inclusive prefix sum with alternating device scratch arrays.
static void inclusive_scan_int_sycl(
    sycl::queue& q,
    const int* d_in,
    int* d_out,
    int* d_tmp,
    int n)
{
    if (n <= 0) return;

    int* original_out = d_out;
    int* current = d_out;
    int* next = d_tmp;

    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        int i = idx[0];
        current[i] = d_in[i];
    }).wait();

    for (int offset = 1; offset < n; offset <<= 1) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            int i = idx[0];
            int val = current[i];

            if (i >= offset) {
                val += current[i - offset];
            }

            next[i] = val;
        }).wait();

        std::swap(current, next);
    }

    if (current != original_out) {
        q.memcpy(original_out, current, sizeof(int) * n).wait();
    }
}

// Shift the inclusive result to obtain exclusive bucket offsets.
static void exclusive_scan_int_sycl(
    sycl::queue& q,
    const int* d_in,
    int* d_out,
    int* d_tmp1,
    int* d_tmp2,
    int n)
{
    if (n <= 0) return;

    inclusive_scan_int_sycl(q, d_in, d_tmp1, d_tmp2, n);

    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        int i = idx[0];

        if (i == 0) {
            d_out[i] = 0;
        } else {
            d_out[i] = d_tmp1[i - 1];
        }
    }).wait();
}

/*---------------- Optional GPU degree ordering ----------------*/
// The main executable uses the host oneDPL sorter instead of this helper.
CSR_Order sortCSRByDegree_SYCL(const CSR& csr, sycl::queue& q)
{
    CSR_Order result;

    if (csr.RowOffset.size() < 2) {
        return result;
    }

    const int num_rows = static_cast<int>(csr.RowOffset.size()) - 1;

    if (num_rows <= 0) {
        return result;
    }

    if (csr.ColInd.size() != csr.Value.size()) {
        throw std::runtime_error("CSR ColInd and Value size mismatch.");
    }

    const int nnz = static_cast<int>(csr.ColInd.size());

    int* d_oldRowOffset = sycl::malloc_device<int>(num_rows + 1, q);
    int* d_oldColInd    = sycl::malloc_device<int>(std::max(1, nnz), q);
    float* d_oldValue   = sycl::malloc_device<float>(std::max(1, nnz), q);

    int* d_degrees = sycl::malloc_device<int>(num_rows, q);

    q.memcpy(
        d_oldRowOffset,
        csr.RowOffset.data(),
        sizeof(int) * (num_rows + 1)
    ).wait();

    if (nnz > 0) {
        q.memcpy(
            d_oldColInd,
            csr.ColInd.data(),
            sizeof(int) * nnz
        ).wait();

        q.memcpy(
            d_oldValue,
            csr.Value.data(),
            sizeof(float) * nnz
        ).wait();
    }

    q.parallel_for(sycl::range<1>(num_rows), [=](sycl::id<1> idx) {
        int row = idx[0];
        d_degrees[row] = d_oldRowOffset[row + 1] - d_oldRowOffset[row];
    }).wait();

    int max_degree = 0;
    for (int i = 0; i < num_rows; ++i) {
        int degree = csr.RowOffset[i + 1] - csr.RowOffset[i];
        if (degree > max_degree) {
            max_degree = degree;
        }
    }

    const int num_bins = max_degree + 1;

    // Count rows by degree, then scan the counts into bucket boundaries.
    int* d_count    = sycl::malloc_device<int>(num_bins, q);
    int* d_binStart = sycl::malloc_device<int>(num_bins + 1, q);
    int* d_writePos = sycl::malloc_device<int>(num_bins, q);

    q.memset(d_count, 0, sizeof(int) * num_bins).wait();

    q.parallel_for(sycl::range<1>(num_rows), [=](sycl::id<1> idx) {
        int row = idx[0];
        int degree = d_degrees[row];

        sycl::atomic_ref<
            int,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space
        > atomic_count(d_count[degree]);

        atomic_count.fetch_add(1);
    }).wait();

    int scan_size = std::max(num_bins, num_rows);
    int* d_scanTmp1 = sycl::malloc_device<int>(scan_size, q);
    int* d_scanTmp2 = sycl::malloc_device<int>(scan_size, q);

    exclusive_scan_int_sycl(
        q,
        d_count,
        d_binStart,
        d_scanTmp1,
        d_scanTmp2,
        num_bins
    );

    int host_num_rows = num_rows;
    q.memcpy(
        d_binStart + num_bins,
        &host_num_rows,
        sizeof(int)
    ).wait();

    q.memcpy(
        d_writePos,
        d_binStart,
        sizeof(int) * num_bins
    ).wait();

    // Scatter row IDs into degree buckets; ties need not keep their input order.
    int* d_originLocal   = sycl::malloc_device<int>(num_rows, q);
    int* d_sortedDegrees = sycl::malloc_device<int>(num_rows, q);

    q.parallel_for(sycl::range<1>(num_rows), [=](sycl::id<1> idx) {
        int row = idx[0];
        int degree = d_degrees[row];

        sycl::atomic_ref<
            int,
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space
        > atomic_pos(d_writePos[degree]);

        int pos = atomic_pos.fetch_add(1);

        d_originLocal[pos] = row;
        d_sortedDegrees[pos] = degree;
    }).wait();

    // Scan the reordered degrees before copying each row's sparse entries.
    int* d_newRowOffset = sycl::malloc_device<int>(num_rows + 1, q);

    int zero = 0;
    q.memcpy(d_newRowOffset, &zero, sizeof(int)).wait();

    inclusive_scan_int_sycl(
        q,
        d_sortedDegrees,
        d_scanTmp1,
        d_scanTmp2,
        num_rows
    );

    q.parallel_for(sycl::range<1>(num_rows), [=](sycl::id<1> idx) {
        int i = idx[0];
        d_newRowOffset[i + 1] = d_scanTmp1[i];
    }).wait();

    int* d_newColInd  = sycl::malloc_device<int>(std::max(1, nnz), q);
    float* d_newValue = sycl::malloc_device<float>(std::max(1, nnz), q);

    if (nnz > 0) {
        constexpr int WG_SIZE = 256;

        sycl::range<1> global_range(static_cast<size_t>(num_rows) * WG_SIZE);
        sycl::range<1> local_range(WG_SIZE);

        q.parallel_for(
            sycl::nd_range<1>(global_range, local_range),
            [=](sycl::nd_item<1> item) {
                int sorted_row = item.get_group(0);
                int local_id = item.get_local_id(0);

                int original_row = d_originLocal[sorted_row];

                int old_start = d_oldRowOffset[original_row];
                int old_end   = d_oldRowOffset[original_row + 1];

                int new_start = d_newRowOffset[sorted_row];

                int degree = old_end - old_start;

                for (int j = local_id; j < degree; j += WG_SIZE) {
                    int old_idx = old_start + j;
                    int new_idx = new_start + j;

                    d_newColInd[new_idx] = d_oldColInd[old_idx];
                    d_newValue[new_idx]  = d_oldValue[old_idx];
                }
            }
        ).wait();
    }

    result.Origin_local.resize(num_rows);
    result.RowOffset.resize(num_rows + 1);
    result.ColInd.resize(nnz);
    result.Value.resize(nnz);

    q.memcpy(
        result.Origin_local.data(),
        d_originLocal,
        sizeof(int) * num_rows
    ).wait();

    q.memcpy(
        result.RowOffset.data(),
        d_newRowOffset,
        sizeof(int) * (num_rows + 1)
    ).wait();

    if (nnz > 0) {
        q.memcpy(
            result.ColInd.data(),
            d_newColInd,
            sizeof(int) * nnz
        ).wait();

        q.memcpy(
            result.Value.data(),
            d_newValue,
            sizeof(float) * nnz
        ).wait();
    }

    sycl::free(d_oldRowOffset, q);
    sycl::free(d_oldColInd, q);
    sycl::free(d_oldValue, q);

    sycl::free(d_degrees, q);
    sycl::free(d_count, q);
    sycl::free(d_binStart, q);
    sycl::free(d_writePos, q);

    sycl::free(d_originLocal, q);
    sycl::free(d_sortedDegrees, q);
    sycl::free(d_newRowOffset, q);
    sycl::free(d_newColInd, q);
    sycl::free(d_newValue, q);

    sycl::free(d_scanTmp1, q);
    sycl::free(d_scanTmp2, q);

    return result;
}

/*---------------- Device operation-cost calibration ----------------*/
SpmmCostParams measure_spmm_costs(
    sycl::queue& q,
    int K,
    int repeat,
    int WI,
    int data_MB_for_load,
    int atomic_target_writers,
    const std::vector<int>& atomic_group_candidates)
{
    SpmmCostParams out{};
    out.K = K;
    out.WI = WI;
    out.repeat = repeat;

    // Calibration has its own warm-up count, independent of the SpMM driver.
    const int warmup = 20;
    const bool profiling_enabled = q.has_property<sycl::property::queue::enable_profiling>();

    (void)atomic_group_candidates;

    auto wait_event_or_log = [&](sycl::event ev, const char* tag) {
        ev.wait();
        std::cout << "[OK] " << tag << "\n";
    };

    std::cout << "[measure_spmm_costs] begin\n";

    // ---------------- 1) Compute ----------------
    {
        std::cout << "[Compute] begin\n";

        float* sink = nullptr;
        try {
            sink = sycl::malloc_shared<float>(static_cast<size_t>(WI), q);
            if (!sink) throw std::bad_alloc();
        } catch (...) {
            if (sink) sycl::free(sink, q);
            throw;
        }

        // warmup
        for (int w = 0; w < warmup; ++w) {
            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for<class bench_compute_warm2>(sycl::range<1>(WI), [=](sycl::id<1> id) {
                    float acc = 0.0f;
                    float a = 1.001f, b = 0.999f;
                    for (int i = 0; i < K; ++i) acc = sycl::fma(acc, a, b);
                    sink[id[0]] = acc;
                });
            });

            if (w == 0) wait_event_or_log(ev, "compute warmup(0)");
            else ev.wait();
        }

        // timed
        double total_ms = 0.0;
        for (int r = 0; r < repeat; ++r) {
            std::chrono::steady_clock::time_point t0, t1;
            if (!profiling_enabled) t0 = std::chrono::steady_clock::now();

            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for<class bench_compute2>(sycl::range<1>(WI), [=](sycl::id<1> id) {
                    float acc = 0.0f;
                    float a = 1.001f, b = 0.999f;
                    for (int i = 0; i < K; ++i) acc = sycl::fma(acc, a, b);
                    sink[id[0]] = acc;
                });
            });

            if (r == 0) wait_event_or_log(ev, "compute timed(0)");
            else ev.wait();

            if (!profiling_enabled) t1 = std::chrono::steady_clock::now();
            total_ms += event_elapsed_ms_or_fallback(ev, profiling_enabled, t0, t1);
        }

        const double avg_ms = total_ms / double(repeat);
        const double avg_ns_per_wi = (avg_ms * 1e6) / double(WI);
        out.C_compute_ns = avg_ns_per_wi / double(K) * 10;

        sycl::free(sink, q);

        q.wait();
        std::cout << "[Compute] end\n";
    }

    // ---------------- 2) Load (random rows, repeated feature accesses) ----------------
    {
        std::cout << "[Load] begin\n";

        constexpr int SG_SIZE = 32;

        const size_t total_bytes  = static_cast<size_t>(data_MB_for_load) * 1024ull * 1024ull;
        const size_t total_floats = total_bytes / sizeof(float);
        if (total_floats < static_cast<size_t>(K)) {
            throw std::runtime_error("data_MB_for_load too small for K.");
        }
        const size_t data_rows = total_floats / static_cast<size_t>(K);

        float* data = nullptr;
        uint32_t* idx_dev = nullptr;
        try {
            data = sycl::malloc_shared<float>(total_floats, q);
            if (!data) throw std::bad_alloc();
            idx_dev = sycl::malloc_shared<uint32_t>(WI, q);
            if (!idx_dev) throw std::bad_alloc();
        } catch (...) {
            if (data) sycl::free(data, q);
            if (idx_dev) sycl::free(idx_dev, q);
            throw;
        }

        for (size_t i = 0; i < total_floats; ++i) data[i] = 1.2345f;
        {
            std::mt19937_64 rng(123456789ULL);
            std::uniform_int_distribution<uint32_t> dist(0U, static_cast<uint32_t>(data_rows - 1));
            for (int g = 0; g < WI; g += SG_SIZE) {
                const uint32_t row = dist(rng);
                for (int t = 0; t < SG_SIZE && (g + t) < WI; ++t) idx_dev[g + t] = row;
            }
        }

        // warmup
        for (int w = 0; w < warmup; ++w) {
            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for<class bench_load_warm2>(sycl::range<1>(WI), [=](sycl::id<1> id) {
                    const size_t gid  = id[0];
                    const int lane = static_cast<int>(gid % SG_SIZE);
                    const size_t group_base = gid - static_cast<size_t>(lane);
                    const uint32_t row = idx_dev[group_base];

                    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(K);

                    volatile float s = 0.0f;
                    for (int t = 0; t < K; ++t) {
                        int k = t * SG_SIZE + lane;
                        if (k >= K) k %= K;
                        s += data[base + static_cast<size_t>(k)];
                    }
                    (void)s;
                });
            });

            if (w == 0) wait_event_or_log(ev, "load warmup(0)");
            else ev.wait();
        }

        // timed
        double total_ms = 0.0;
        for (int r = 0; r < repeat; ++r) {
            std::chrono::steady_clock::time_point t0, t1;
            if (!profiling_enabled) t0 = std::chrono::steady_clock::now();

            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for<class bench_load2>(sycl::range<1>(WI), [=](sycl::id<1> id) {
                    const size_t gid  = id[0];
                    const int lane = static_cast<int>(gid % SG_SIZE);
                    const size_t group_base = gid - static_cast<size_t>(lane);
                    const uint32_t row = idx_dev[group_base];

                    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(K);

                    volatile float s = 0.0f;
                    for (int t = 0; t < K; ++t) {
                        int k = t * SG_SIZE + lane;
                        if (k >= K) k %= K;
                        s += data[base + static_cast<size_t>(k)];
                    }
                    (void)s;
                });
            });

            if (r == 0) wait_event_or_log(ev, "load timed(0)");
            else ev.wait();

            if (!profiling_enabled) t1 = std::chrono::steady_clock::now();
            total_ms += event_elapsed_ms_or_fallback(ev, profiling_enabled, t0, t1);
        }

        const double avg_ms = total_ms / double(repeat);
        const double avg_ns_per_wi = (avg_ms * 1e6) / double(WI);
        out.C_load_ns = avg_ns_per_wi / double(K) * 10;

        sycl::free(idx_dev, q);
        sycl::free(data, q);

        q.wait();
        std::cout << "[Load] end\n";
    }

    // ---------------- 3) Store ----------------
    {
        std::cout << "[Store] begin\n";

        float* outbuf = nullptr;
        try {
            outbuf = sycl::malloc_shared<float>(static_cast<size_t>(WI) * static_cast<size_t>(K), q);
            if (!outbuf) throw std::bad_alloc();
        } catch (...) {
            if (outbuf) sycl::free(outbuf, q);
            throw;
        }

        // warmup
        for (int w = 0; w < warmup; ++w) {
            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for<class bench_store_warm2>(sycl::range<1>(WI), [=](sycl::id<1> id) {
                    const size_t tid = id[0];
                    for (int i = 0; i < K; ++i) {
                        outbuf[static_cast<size_t>(i) * static_cast<size_t>(WI) + tid] = float(i);
                    }
                });
            });

            if (w == 0) wait_event_or_log(ev, "store warmup(0)");
            else ev.wait();
        }

        // timed
        double total_ms = 0.0;
        for (int r = 0; r < repeat; ++r) {
            std::chrono::steady_clock::time_point t0, t1;
            if (!profiling_enabled) t0 = std::chrono::steady_clock::now();

            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for<class bench_store2>(sycl::range<1>(WI), [=](sycl::id<1> id) {
                    const size_t tid = id[0];
                    for (int i = 0; i < K; ++i) {
                        outbuf[static_cast<size_t>(i) * static_cast<size_t>(WI) + tid] = float(i);
                    }
                });
            });

            if (r == 0) wait_event_or_log(ev, "store timed(0)");
            else ev.wait();

            if (!profiling_enabled) t1 = std::chrono::steady_clock::now();
            total_ms += event_elapsed_ms_or_fallback(ev, profiling_enabled, t0, t1);
        }

        const double avg_ms = total_ms / double(repeat);
        const double avg_ns_per_wi = (avg_ms * 1e6) / double(WI);
        out.C_store_ns = avg_ns_per_wi / double(K) * 10;

        sycl::free(outbuf, q);

        q.wait();
        std::cout << "[Store] end\n";
    }

    // ---------------- 4) Atomic
    // SpMM-aligned: float atomic_ref, relaxed order, device scope
    // Report the normalized per-atomic-operation weight.
    // ----------------
    {
        std::cout << "[Atomic] begin\n";

        // Number of logical writer groups mapped to the same output row.
        // This parameter represents structural writer multiplicity rather
        // than the number of work-groups guaranteed to execute simultaneously.
        const int groups_per_row = std::max(1, atomic_target_writers);

        // This benchmark logically treats every 128 work-items as one group.
        const int pseudo_local_size = 128;

        // Derive a logical feature block from K, rounded down to a multiple of 32
        // and restricted to the range [32, 128].
        int K_block = std::min(K, 128);
        K_block = (K_block / 32) * 32;
        if (K_block <= 0) {
            K_block = 32;
        }

        const int atomic_kk_per_group = K_block;

        const int total_groups_target =
            std::max(1, (WI + pseudo_local_size - 1) / pseudo_local_size);

        // G is the number of independent logical output rows.
        int G = std::max(1, total_groups_target / groups_per_row);

        int total_groups = G * groups_per_row;
        while (total_groups < total_groups_target) {
            ++G;
            total_groups = G * groups_per_row;
        }

        const size_t global_threads =
            static_cast<size_t>(total_groups) *
            static_cast<size_t>(pseudo_local_size);

        // counters layout: [G][K_block]
        const size_t counter_len =
            static_cast<size_t>(G) *
            static_cast<size_t>(K_block);

        float* counters = nullptr;

        try {
            counters = sycl::malloc_device<float>(counter_len, q);

            if (!counters) {
                throw std::bad_alloc();
            }
        } catch (...) {
            if (counters) {
                sycl::free(counters, q);
            }
            throw;
        }

        auto zero_counters = [&]() {
            q.fill(counters, 0.0f, counter_len).wait();
        };

        zero_counters();

        // ---------------- Warm-up ----------------
        for (int w = 0; w < warmup; ++w) {
            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for<class bench_atomic_spmm_f32_warm>(
                    sycl::range<1>(global_threads),
                    [=](sycl::id<1> tid) {
                        const uint32_t t = static_cast<uint32_t>(tid[0]);

                        const uint32_t lane = t % static_cast<uint32_t>(pseudo_local_size);

                        const uint32_t group_id = t / static_cast<uint32_t>(pseudo_local_size);

                        // groups_per_row pseudo groups update the same row.
                        const uint32_t row_gid = group_id / static_cast<uint32_t>(groups_per_row);

                        if (row_gid >= static_cast<uint32_t>(G)) {
                            return;
                        }

                        for (uint32_t kk = lane;
                            kk < static_cast<uint32_t>(atomic_kk_per_group);
                            kk += static_cast<uint32_t>(pseudo_local_size)) {

                            const size_t idx =
                                static_cast<size_t>(row_gid) * static_cast<size_t>(K_block) + 
                                static_cast<size_t>(kk);

                            if (idx >= counter_len) {
                                return;
                            }

                            float* ptr = counters + idx;

                            sycl::atomic_ref<
                                float,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                atomic_value(*ptr);

                            atomic_value.fetch_add(1.0f);
                        }
                    });
            });

            if (w == 0) {
                wait_event_or_log(ev, "atomic float warmup(0)");
            } else {
                ev.wait();
            }

            zero_counters();
        }

        // ---------------- Timed runs ----------------
        double total_ms = 0.0;

        for (int r = 0; r < repeat; ++r) {
            zero_counters();

            std::chrono::steady_clock::time_point t0;
            std::chrono::steady_clock::time_point t1;

            if (!profiling_enabled) {
                t0 = std::chrono::steady_clock::now();
            }

            auto ev = q.submit([&](sycl::handler& h) {
                h.parallel_for<class bench_atomic_spmm_f32>(
                    sycl::range<1>(global_threads),
                    [=](sycl::id<1> tid) {
                        const uint32_t t =
                            static_cast<uint32_t>(tid[0]);

                        const uint32_t lane =
                            t % static_cast<uint32_t>(pseudo_local_size);

                        const uint32_t group_id =
                            t / static_cast<uint32_t>(pseudo_local_size);

                        const uint32_t row_gid =
                            group_id /
                            static_cast<uint32_t>(groups_per_row);

                        if (row_gid >= static_cast<uint32_t>(G)) {
                            return;
                        }

                        for (uint32_t kk = lane;
                            kk < static_cast<uint32_t>(atomic_kk_per_group);
                            kk += static_cast<uint32_t>(pseudo_local_size)) {

                            const size_t idx =
                                static_cast<size_t>(row_gid) * static_cast<size_t>(K_block) +
                                static_cast<size_t>(kk);

                            if (idx >= counter_len) {
                                return;
                            }

                            float* ptr = counters + idx;

                            sycl::atomic_ref<
                                float,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>
                                atomic_value(*ptr);

                            atomic_value.fetch_add(1.0f);
                        }
                    });
            });

            if (r == 0) {
                wait_event_or_log(ev, "atomic float timed(0)");
            } else {
                ev.wait();
            }

            if (!profiling_enabled) {
                t1 = std::chrono::steady_clock::now();
            }

            total_ms += event_elapsed_ms_or_fallback(
                ev,
                profiling_enabled,
                t0,
                t1);
        }

        const double avg_ms =
            total_ms / static_cast<double>(repeat);

        /*
        * Each pseudo group performs one atomic operation for every
        * feature position in [0, atomic_kk_per_group).
        */
        const double total_atomic_ops =
            static_cast<double>(total_groups) *
            static_cast<double>(atomic_kk_per_group);

        const double avg_ns_per_atomic =
            (avg_ms * 1.0e6) / total_atomic_ops;

        // Apply the same factor-of-ten normalization as compute, load and store.
        out.C_atomic_ns = avg_ns_per_atomic * 10.0;
        out.atomic_writers_per_addr = groups_per_row;

        std::cout
            << "[Atomic] data_type=float"
            << " groups_per_row=" << groups_per_row
            << " independent_rows=" << G
            << " total_groups=" << total_groups
            << " K_block=" << K_block
            << " total_atomic_ops=" << total_atomic_ops
            << " avg_ms=" << avg_ms
            << " ns_per_atomic=" << avg_ns_per_atomic
            << "\n";

        q.wait();
        sycl::free(counters, q);
        q.wait();

        std::cout << "[Atomic] end\n";
    }

    std::cout << "[measure_spmm_costs] end\n";
    return out;
}

/*---------------- Cost-weighted task partitioning ----------------*/
std::vector<ParallelDataWeighted> costWeightedTaskPartitioning(
    const CSR_Order &csr,
    const SpmmCostParams &costs,
    double capacity_weight)
{
    std::vector<ParallelDataWeighted> tasks;

    const auto &rp = csr.RowOffset;
    const int n = static_cast<int>(rp.size()) - 1;
    if (n <= 0) return tasks;

    const double Cc = costs.C_compute_ns;
    const double Cl = costs.C_load_ns;
    const double Cs = costs.C_store_ns;
    const double Ca = costs.C_atomic_ns;
    const double wCL = Cc + Cl;

    // Complete rows use direct stores; partial fragments use atomic updates.
    auto full_w = [&](int r) -> double {
        return wCL * static_cast<double>(rp[r + 1] - rp[r]) + Cs;
    };

    auto chunk_w = [&](int nnz) -> double {
        return wCL * static_cast<double>(nnz) + Ca;
    };

    // Recover the segment length represented by the selected weighted capacity.
    const int S = std::max(
        1,
        static_cast<int>(
            std::floor((capacity_weight - Ca) / std::max(1e-30, wCL))
        )
    );

    // Store a compact task descriptor instead of copying its sparse entries.
    auto emit = [&](int s, int e, int fo, int last) {
        if (s < e) {
            tasks.push_back({s, e, fo, last});
        }
    };

    int row = 0;
    int g_start = 0;
    int first_off = 0;
    double acc = 0.0;

    // Fill residual capacity with the current long row's degree % S prefix.
    auto try_absorb_current_prefix =
        [&](int row_deg, int row_idx, int &remaining, int &off) -> void
    {
        if (acc <= 0.0) return;

        const int r = row_deg % S;

        if (r == 0 || r >= row_deg) {
            emit(g_start, row_idx, first_off, 0);
            g_start = row_idx;
            first_off = 0;
            acc = 0.0;
            return;
        }

        const double w = chunk_w(r);
        const double rem = capacity_weight - acc;

        if (w <= rem) {
            emit(g_start, row_idx + 1, first_off, r);

            g_start = row_idx;
            first_off = r;
            acc = 0.0;

            remaining -= r;
            off += r;
        } else {
            emit(g_start, row_idx, first_off, 0);
            g_start = row_idx;
            first_off = 0;
            acc = 0.0;
        }
    };

    // Let the final fragment absorb the next long row's prefix when it fits.
    auto try_absorb_next_prefix = [&](int cur_row) -> bool {
        if (acc >= capacity_weight) return false;
        if (cur_row + 1 >= n) return false;

        const int nr = cur_row + 1;
        const int nd = rp[nr + 1] - rp[nr];

        if (full_w(nr) <= capacity_weight) return false;

        const int r2 = nd % S;

        if (r2 == 0 || r2 >= nd) return false;

        const double w = chunk_w(r2);
        const double rem = capacity_weight - acc;
        if (w > rem) return false;

        emit(g_start, nr + 1, first_off, r2);

        g_start = nr;
        first_off = r2;
        acc = 0.0;

        return true;
    };

    while (row < n) {
        const int deg = rp[row + 1] - rp[row];
        const double wr = full_w(row);

        // Greedily pack complete rows without splitting them.
        if (wr <= capacity_weight) {
            if (acc == 0.0 || acc + wr <= capacity_weight) {
                acc += wr;
                ++row;
                continue;
            }

            emit(g_start, row, first_off, 0);
            g_start = row;
            first_off = 0;
            acc = 0.0;
            continue;
        }

        int off = 0;

        // A previous task may already own the beginning of this row.
        const bool has_absorbed_prefix =
            (acc == 0.0 && g_start == row && first_off > 0);

        if (has_absorbed_prefix) {
            off = first_off;
        }

        int remaining = deg - off;

        if (remaining <= 0) {
            g_start = row + 1;
            first_off = 0;
            acc = 0.0;
            ++row;
            continue;
        }

        if (!has_absorbed_prefix) {
            try_absorb_current_prefix(deg, row, remaining, off);
        }

        // Split the remaining long row into segments of at most S nonzeros.
        while (remaining > 0) {
            const int take = std::min(remaining, S);
            const double w = chunk_w(take);

            acc += w;
            remaining -= take;
            off += take;

            if (remaining > 0) {
                emit(g_start, row + 1, first_off, off);

                g_start = row;
                first_off = off;
                acc = 0.0;
                continue;
            }

            if (remaining == 0) {
                if (!try_absorb_next_prefix(row)) {
                    emit(g_start, row + 1, first_off, 0);
                    g_start = row + 1;
                    first_off = 0;
                    acc = 0.0;
                }
            }
        }

        ++row;
    }

    // Flush any rows or fragments still held in the current task.
    if (acc > 0.0 || g_start < n) {
        emit(g_start, n, first_off, 0);
    }

    return tasks;
}

/*---------------- Cost-file input/output ----------------*/
bool save_costs_one_per_line(
    SpmmCostParams p,
    const std::string& filepath,
    bool append,
    int precision)
{
    namespace fs = std::filesystem;

    std::error_code ec;
    fs::path path(filepath);

    if (filepath.empty()) {
        return false;
    }

    if (path.has_parent_path()) {
        fs::path dir = path.parent_path();

        if (fs::exists(dir, ec) && !fs::is_directory(dir, ec)) {
            return false;
        }

        if (!fs::exists(dir, ec)) {
            if (!fs::create_directories(dir, ec) && ec) {
                return false;
            }
        }
    }

    std::ios::openmode mode = std::ios::out;

    if (append) {
        mode |= std::ios::app;
    } else {
        mode |= std::ios::trunc;
    }

    std::ofstream ofs(path, mode);

    if (!ofs.is_open()) {
        return false;
    }

    // Keep the file order consistent with read_costs_one_per_line().
    ofs << std::fixed << std::setprecision(precision)
        << p.C_compute_ns << "\n"
        << p.C_load_ns    << "\n"
        << p.C_store_ns   << "\n"
        << p.C_atomic_ns  << "\n";

    return ofs.good();
}

bool read_costs_one_per_line(SpmmCostParams& out, const std::string& filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) return false;

    double v1, v2, v3, v4;
    if (!(ifs >> v1)) return false;
    if (!(ifs >> v2)) return false;
    if (!(ifs >> v3)) return false;
    if (!(ifs >> v4)) return false;

    out.C_compute_ns = v1;
    out.C_load_ns    = v2;
    out.C_store_ns   = v3;
    out.C_atomic_ns  = v4;
    return true;
}

/*---------------- Caller-supplied reference validation ----------------*/
std::vector<std::pair<int, float>> get_csr_row_elements(
    const CSR& csr,
    int row)
{
    std::vector<std::pair<int, float>> elements;

    const int row_count = static_cast<int>(csr.RowOffset.size()) - 1;
    if (row < 0 || row >= row_count) {
        return elements;
    }

    const int begin = csr.RowOffset[row];
    const int end = csr.RowOffset[row + 1];
    elements.reserve(static_cast<std::size_t>(std::max(0, end - begin)));

    for (int index = begin; index < end; ++index) {
        elements.emplace_back(csr.ColInd[index], csr.Value[index]);
    }

    return elements;
}

bool validate_custom_csr_result(
    const Defined::RowMajorMatrixXf& sycl_result,
    const Defined::RowMajorMatrixXf& reference,
    const CSR& csr,
    const Defined::RowMajorMatrixXf& dense_matrix,
    float abs_tolerance)
{
    try {
        if (abs_tolerance < 0.0f || !std::isfinite(abs_tolerance)) {
            throw std::invalid_argument(
                "Validation tolerance must be finite and non-negative.");
        }

        if (sycl_result.rows() != reference.rows() ||
            sycl_result.cols() != reference.cols()) {
            std::cerr << "Validation failed: result matrix shape is ("
                      << sycl_result.rows() << ", " << sycl_result.cols()
                      << "), but reference matrix shape is ("
                      << reference.rows() << ", " << reference.cols()
                      << ").\n";
            return false;
        }

        if (csr.RowOffset.size() !=
            static_cast<std::size_t>(reference.rows()) + 1) {
            std::cerr << "Validation failed: CSR row count does not match the reference matrix.\n";
            return false;
        }

        if (dense_matrix.rows() <= 0 ||
            dense_matrix.cols() != reference.cols()) {
            std::cerr << "Validation failed: input dense matrix shape is incompatible with the reference result.\n";
            return false;
        }

        std::vector<ErrorElement> error_elements;
        error_elements.reserve(10);

        for (int row = 0;
             row < reference.rows() && error_elements.size() < 10;
             ++row) {
            for (int col = 0;
                 col < reference.cols() && error_elements.size() < 10;
                 ++col) {
                const float sycl_value = sycl_result(row, col);
                const float reference_value = reference(row, col);

                bool mismatch = false;
                float absolute_error = 0.0f;

                if (std::isnan(sycl_value) || std::isnan(reference_value)) {
                    mismatch = !(std::isnan(sycl_value) &&
                                 std::isnan(reference_value));
                    absolute_error = mismatch
                        ? std::numeric_limits<float>::infinity()
                        : 0.0f;
                } else if (std::isinf(sycl_value) ||
                           std::isinf(reference_value)) {
                    mismatch = sycl_value != reference_value;
                    absolute_error = mismatch
                        ? std::numeric_limits<float>::infinity()
                        : 0.0f;
                } else {
                    absolute_error = std::abs(sycl_value - reference_value);
                    mismatch = absolute_error > abs_tolerance;
                }

                if (!mismatch) {
                    continue;
                }

                ErrorElement error;
                error.result_row = row;
                error.result_col = col;
                error.sycl_value = sycl_value;
                error.reference_value = reference_value;
                error.absolute_error = absolute_error;
                error.csr_row_elements = get_csr_row_elements(csr, row);

                error.dense_col_elements.resize(dense_matrix.rows());
                for (int dense_row = 0;
                     dense_row < dense_matrix.rows();
                     ++dense_row) {
                    error.dense_col_elements[dense_row] =
                        dense_matrix(dense_row, col);
                }

                error_elements.push_back(std::move(error));
            }
        }

        if (error_elements.empty()) {
            std::cout
                << "\033[1;32mValidation passed: all elements satisfy the comparison criteria.\033[0m\n";
            return true;
        }

        std::cout << "\033[1;31mValidation failed: showing "
                  << error_elements.size()
                  << " mismatched elements (up to the first 10).\033[0m\n";

        for (std::size_t index = 0;
             index < error_elements.size();
             ++index) {
            const ErrorElement& error = error_elements[index];

            std::cout << "\nStored entries in CSR row " << error.result_row
                      << ":\n";
            std::cout << "RowOffset[" << error.result_row << "]: "
                      << csr.RowOffset[error.result_row] << '\n';
            std::cout << "RowOffset[" << error.result_row + 1 << "]: "
                      << csr.RowOffset[error.result_row + 1] << '\n';
            std::cout << "Format: [ColInd] -> Value\n";

            for (const auto& element : error.csr_row_elements) {
                std::cout << "  [" << element.first << "] -> "
                          << element.second << '\n';
            }

            std::cout << "\nDense matrix column " << error.result_col
                      << " (up to the first 5 rows):\n";
            const int display_count = std::min(
                5,
                static_cast<int>(error.dense_col_elements.size()));

            for (int dense_row = 0;
                 dense_row < display_count;
                 ++dense_row) {
                std::cout << "  Row " << dense_row << ": "
                          << error.dense_col_elements[dense_row] << '\n';
            }

            std::cout << "\n\033[1;34mMismatch #" << index + 1
                      << ":\033[0m\n";
            std::cout << "Result matrix position: ("
                      << error.result_row << ", "
                      << error.result_col << ")\n";
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "SYCL result:     " << error.sycl_value << '\n';
            std::cout << "Reference value: " << error.reference_value << '\n';
            std::cout << "Absolute error:  " << error.absolute_error << '\n';
        }

        return false;
    } catch (const std::exception& error) {
        std::cerr << "Validation error: " << error.what() << '\n';
        return false;
    }
}

