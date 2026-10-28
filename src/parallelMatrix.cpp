// SYCL task execution with local-memory CSR tiling and mean aggregation.
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include "parallelMatrix.h"
#include <chrono>

using namespace std;

namespace detail {
    inline std::size_t round_up_multiple(std::size_t value, std::size_t align) {
        return (value + align - 1) / align * align;
    }
    inline std::size_t clamp(std::size_t v, std::size_t lo, std::size_t hi) {
        return std::max(lo, std::min(v, hi));
    }
    inline std::size_t next_pow2(std::size_t v) {
        if (v == 0) return 1;
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        if constexpr (sizeof(std::size_t) == 8) {
            v |= v >> 32;
        }
        return v + 1;
    }

    // Prefer the supported alignment closest to 32; do not force a subgroup size.
    inline std::size_t choose_wg_alignment(const sycl::device& dev) {
        const std::vector<std::size_t> supported =
            dev.get_info<sycl::info::device::sub_group_sizes>();
        std::size_t alignment = 1;
        bool found_nonzero = false;
        std::size_t best_distance =
            std::numeric_limits<std::size_t>::max();

        for (const std::size_t candidate : supported) {
            if (candidate == 0) continue;
            const std::size_t distance =
                candidate > 32 ? candidate - 32 : 32 - candidate;
            if (!found_nonzero || distance < best_distance ||
                (distance == best_distance && candidate < alignment)) {
                alignment = candidate;
                best_distance = distance;
                found_nonzero = true;
            }
        }

        return alignment;
    }
} // namespace detail

// ===================== SPMM (Mean) with AoS-weighted groups =====================
// 1 work-group processes exactly one ParallelDataWeighted entry (a group of rows/row-chunks).
// Mean aggregation: each partial accumulates sum over its nnz chunk and divides by full degree, then atomically adds.
// A full-row (begin==0 && end==deg) is written directly without atomics (unique-writer).

Defined::RowMajorMatrixXf spmmKernel(
    sycl::queue &q,
    const CSR_Order& adjOrder,
    const Defined::RowMajorMatrixXf& H,
    int dim_hint,
    int local_dimension,
    std::vector<ParallelDataWeighted> tasks,
    double &time)
{
    const sycl::device dev = q.get_device();
    const std::size_t H_rows = static_cast<std::size_t>(H.rows());
    const std::size_t K      = static_cast<std::size_t>(H.cols());
    const std::size_t row_count = adjOrder.Origin_local.size();
    const std::size_t device_max_wg =
        dev.get_info<sycl::info::device::max_work_group_size>();
    const std::size_t wg_alignment = detail::choose_wg_alignment(dev);

    if (K == 0) {
        time = 0.0;
        return Defined::RowMajorMatrixXf::Zero(row_count, H.cols());
    }
    if (K > device_max_wg) {
        throw std::invalid_argument(
            "spmmKernel supports only K <= device max work-group size; "
            "use spmmKernelLargeKFallback for larger K.");
    }

    // For the original path, one work-item continues to own one feature.
    // Round the work-group up to the device-selected alignment when legal;
    // padded work-items participate in barriers but do not access B or C.
    const std::size_t alignment_remainder = K % wg_alignment;
    const std::size_t alignment_padding =
        alignment_remainder == 0 ? 0 : wg_alignment - alignment_remainder;
    std::size_t local_size = K;
    if (alignment_padding <= device_max_wg - K) {
        local_size = K + alignment_padding;
    }

    if (tasks.empty()) {
        time = 0.0;
        return Defined::RowMajorMatrixXf::Zero(row_count, H.cols());
    }

    // CSR aliases
    const std::vector<int>&   row_offsets = adjOrder.RowOffset;
    const std::vector<int>&   col_index   = adjOrder.ColInd;
    const std::vector<float>& values      = adjOrder.Value;
    const std::vector<int>&   row_origin  = adjOrder.Origin_local;

    Defined::RowMajorMatrixXf C = Defined::RowMajorMatrixXf::Zero(row_count, H.cols());

    // Buffers
    sycl::buffer<int>    buf_row_offsets(row_offsets.data(), sycl::range<1>(row_offsets.size()));
    sycl::buffer<int>    buf_col_index  (col_index.data(),   sycl::range<1>(col_index.size()));
    sycl::buffer<float>  buf_values     (values.data(),      sycl::range<1>(values.size()));
    sycl::buffer<int>    buf_row_origin (row_origin.data(),  sycl::range<1>(row_origin.size()));
    sycl::buffer<float,2>buf_B          (H.data(), sycl::range<2>(H_rows, K));
    sycl::buffer<float,2>buf_C          (C.data(), sycl::range<2>(row_count, K));
    sycl::buffer<ParallelDataWeighted> buf_tasks(tasks.data(), sycl::range<1>(tasks.size()));

    const std::size_t task_count  = tasks.size();
    const std::size_t global_size = detail::round_up_multiple(task_count * local_size, local_size);

    // CSR entries allocated in local memory; main passes K as local_dimension.
    // LM_CAP must be positive and fit within the device's local-memory limit.
    // Each tile uses at most min(local_size, LM_CAP) entries.
    const int LM_CAP = local_dimension;

    sycl::event e = q.submit([&](sycl::handler& h) {
        // Stage CSR column indices and values for reuse across feature columns.
        sycl::local_accessor<int,   1> loc_cols(LM_CAP, h);
        sycl::local_accessor<float, 1> loc_vals(LM_CAP, h);

        auto acc_row_offsets = buf_row_offsets.get_access<sycl::access::mode::read>(h);
        auto acc_col_index   = buf_col_index  .get_access<sycl::access::mode::read>(h);
        auto acc_values      = buf_values     .get_access<sycl::access::mode::read>(h);
        auto acc_row_origin  = buf_row_origin .get_access<sycl::access::mode::read>(h);
        auto acc_B           = buf_B          .get_access<sycl::access::mode::read>(h);
        auto acc_C           = buf_C          .get_access<sycl::access::mode::read_write>(h);
        auto acc_tasks       = buf_tasks      .get_access<sycl::access::mode::read>(h);

        h.parallel_for<class spmm_weightedwg_kernel_v2_hybrid_local_safe>(
            sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> item) {

                const std::size_t gid = item.get_group_linear_id();
                if (gid >= acc_tasks.get_range()[0]) return;

                // Each descriptor covers a half-open row range with boundary offsets.
                const ParallelDataWeighted gtask = acc_tasks[gid];
                const int row_s     = gtask.row_start;
                const int row_e     = gtask.row_end;
                const int first_off = gtask.first_row_offset;
                const int last_nnz  = gtask.last_row_nnz;

                const int local_id    = item.get_local_id(0);
                const int local_range = item.get_local_range(0);
                const bool valid_feature =
                    static_cast<std::size_t>(local_id) < K;

                for (int r = row_s; r < row_e; ++r) {
                    const int csr_row_begin = acc_row_offsets[r];
                    const int csr_row_end   = acc_row_offsets[r + 1];
                    const int deg           = csr_row_end - csr_row_begin;
                    if (deg <= 0) continue;

                    int begin = (r == row_s) ? first_off : 0;
                    int end   = deg;
                    if (r == row_e - 1 && last_nnz > 0) {
                        end = last_nnz;
                    }
                    begin = std::clamp(begin, 0, deg);
                    end   = std::clamp(end,   begin, deg);
                    if (begin >= end) continue;

                    const int chunk_begin = csr_row_begin + begin;
                    const int chunk_end   = csr_row_begin + end;
                    const int chunk_len   = chunk_end - chunk_begin;
                    const int full_deg    = deg;

                    const int k = local_id;
                    float accum = 0.0f;
                    const int tile_capacity =
                        (local_range < LM_CAP) ? local_range : LM_CAP;

                    for (int tile_base = chunk_begin;
                         tile_base < chunk_end;
                         tile_base += tile_capacity) {
                        int remaining = chunk_end - tile_base;
                        int tile_len =
                            (remaining < tile_capacity)
                                ? remaining
                                : tile_capacity;

                        int t = local_id;
                        if (t < tile_len) {
                            int gidx = tile_base + t;
                            loc_cols[t] = acc_col_index[gidx];
                            loc_vals[t] = acc_values[gidx];
                        }
                        // All work-items must see the staged tile, including padded lanes.
                        item.barrier(sycl::access::fence_space::local_space);

                        if (valid_feature) {
                            for (int t = 0; t < tile_len; ++t) {
                                int col = loc_cols[t];
                                float a = loc_vals[t];
                                accum = sycl::fma(a, acc_B[col][k], accum);
                            }
                        }
                        // Finish reading this tile before any lane overwrites it.
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (valid_feature) {
                        // Normalize by the full row degree, including for partial segments.
                        accum /= static_cast<float>(full_deg);

                        // Restore the output row ID once per fragment and feature.
                        const int out_row = acc_row_origin[r];
                        if (begin == 0 && end == full_deg) {
                            // A complete row has a unique writer.
                            acc_C[out_row][k] = accum;
                        } else {
                            // Split-row contributions can arrive from different work-groups.
                            sycl::atomic_ref<float,
                                             sycl::memory_order::relaxed,
                                             sycl::memory_scope::device,
                                             sycl::access::address_space::global_space>
                                atomic_C(acc_C[out_row][k]);
                            atomic_C.fetch_add(accum);
                        }
                    }
                } // rows
            } // lambda
        ); // parallel_for
    }); // submit

    q.wait();

    // Report kernel command time only; allocations and transfers are excluded.
    double timeA = 0.0;
    try {
        timeA =
            (e.template get_profiling_info<sycl::info::event_profiling::command_end>()
             - e.template get_profiling_info<sycl::info::event_profiling::command_start>());
    } catch (...) {
    }
    time = timeA / 1e3;

    return C;
}

// ===================== Large-K fallback for the CWLB SpMM kernel =====================
//
// This function is intentionally separate from spmmKernel().  It preserves the
// original mapping from one ParallelDataWeighted task to one work-group, the CSR
// tile staged in local memory, mean normalization, and direct/atomic output
// semantics. Work-items can process multiple feature columns when K exceeds
// the device's maximum work-group size. The local-memory tile is bounded
// independently of K.
Defined::RowMajorMatrixXf spmmKernelLargeKFallback(
    sycl::queue &q,
    const CSR_Order& adjOrder,
    const Defined::RowMajorMatrixXf& H,
    int dim_hint,
    int local_dimension,
    std::vector<ParallelDataWeighted> tasks,
    double &time)
{
    (void)dim_hint;
    (void)local_dimension;

    const sycl::device dev = q.get_device();
    const std::size_t device_max_wg =
        dev.get_info<sycl::info::device::max_work_group_size>();
    const std::size_t H_rows = static_cast<std::size_t>(H.rows());
    const std::size_t K = static_cast<std::size_t>(H.cols());
    const std::size_t row_count = adjOrder.Origin_local.size();

    if (device_max_wg == 0) {
        throw std::runtime_error(
            "Large-K fallback cannot use a device with max_work_group_size=0.");
    }
    if (K <= device_max_wg) {
        throw std::invalid_argument(
            "spmmKernelLargeKFallback requires K > device max work-group size.");
    }
    if (tasks.empty()) {
        time = 0.0;
        return Defined::RowMajorMatrixXf::Zero(row_count, H.cols());
    }

    // The selected value is only a work-group alignment.  The fallback does
    // not assume that it equals the runtime subgroup size used by the kernel.
    const std::size_t wg_alignment = detail::choose_wg_alignment(dev);

    const std::size_t aligned_max_wg =
        (device_max_wg / wg_alignment) * wg_alignment;
    if (aligned_max_wg == 0) {
        throw std::runtime_error(
            "Large-K fallback cannot form a nonzero aligned work-group size.");
    }

    const auto ceil_div = [](std::size_t numerator, std::size_t denominator) {
        if (denominator == 0) {
            throw std::invalid_argument(
                "Large-K fallback ceil division received a zero denominator.");
        }
        return numerator / denominator +
               static_cast<std::size_t>(numerator % denominator != 0);
    };

    // Cover K in multiple feature rounds using a legal, aligned work-group.
    const std::size_t feature_rounds = ceil_div(K, aligned_max_wg);
    const std::size_t features_per_round = ceil_div(K, feature_rounds);
    const std::size_t aligned_units =
        ceil_div(features_per_round, wg_alignment);
    if (aligned_units >
        std::numeric_limits<std::size_t>::max() / wg_alignment) {
        throw std::overflow_error(
            "Large-K fallback work-group size calculation overflowed.");
    }
    const std::size_t local_size = aligned_units * wg_alignment;

    if (feature_rounds < 2 || local_size == 0 ||
        local_size > aligned_max_wg || local_size > device_max_wg ||
        local_size % wg_alignment != 0 ||
        local_size < ceil_div(K, feature_rounds)) {
        throw std::runtime_error(
            "Large-K fallback derived an invalid work-group configuration.");
    }
    if (local_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "Large-K fallback work-group size exceeds the kernel index range.");
    }
    if (tasks.size() >
        std::numeric_limits<std::size_t>::max() / local_size) {
        throw std::overflow_error(
            "Large-K fallback global work size calculation overflowed.");
    }

    // Keep the original local-memory role (CSR column/value staging), but do
    // not size the tile by K.  That would make local allocation grow without
    // bound precisely in the Large-K case.  The tile is capped by both the
    // work-group size and the device-reported local-memory capacity.
    const std::size_t local_mem_bytes =
        dev.get_info<sycl::info::device::local_mem_size>();
    const std::size_t bytes_per_entry = sizeof(int) + sizeof(float);
    const std::size_t max_local_entries = local_mem_bytes / bytes_per_entry;
    const std::size_t local_capacity =
        std::min(local_size, max_local_entries);
    if (local_capacity == 0 ||
        local_capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "Large-K fallback has insufficient device local memory for one CSR entry.");
    }

    const std::vector<int>& row_offsets = adjOrder.RowOffset;
    const std::vector<int>& col_index = adjOrder.ColInd;
    const std::vector<float>& values = adjOrder.Value;
    const std::vector<int>& row_origin = adjOrder.Origin_local;

    Defined::RowMajorMatrixXf C =
        Defined::RowMajorMatrixXf::Zero(row_count, H.cols());

    sycl::buffer<int> buf_row_offsets(
        row_offsets.data(), sycl::range<1>(row_offsets.size()));
    sycl::buffer<int> buf_col_index(
        col_index.data(), sycl::range<1>(col_index.size()));
    sycl::buffer<float> buf_values(
        values.data(), sycl::range<1>(values.size()));
    sycl::buffer<int> buf_row_origin(
        row_origin.data(), sycl::range<1>(row_origin.size()));
    sycl::buffer<float, 2> buf_B(
        H.data(), sycl::range<2>(H_rows, K));
    sycl::buffer<float, 2> buf_C(
        C.data(), sycl::range<2>(row_count, K));
    sycl::buffer<ParallelDataWeighted> buf_tasks(
        tasks.data(), sycl::range<1>(tasks.size()));

    const std::size_t task_count = tasks.size();
    const std::size_t global_size = task_count * local_size;

    sycl::event e = q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> loc_cols(local_capacity, h);
        sycl::local_accessor<float, 1> loc_vals(local_capacity, h);

        auto acc_row_offsets =
            buf_row_offsets.get_access<sycl::access::mode::read>(h);
        auto acc_col_index =
            buf_col_index.get_access<sycl::access::mode::read>(h);
        auto acc_values =
            buf_values.get_access<sycl::access::mode::read>(h);
        auto acc_row_origin =
            buf_row_origin.get_access<sycl::access::mode::read>(h);
        auto acc_B = buf_B.get_access<sycl::access::mode::read>(h);
        auto acc_C = buf_C.get_access<sycl::access::mode::read_write>(h);
        auto acc_tasks = buf_tasks.get_access<sycl::access::mode::read>(h);

        h.parallel_for<class spmm_weightedwg_large_k_fallback_kernel>(
            sycl::nd_range<1>(
                sycl::range<1>(global_size),
                sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> item) {
                const std::size_t gid = item.get_group_linear_id();
                if (gid >= acc_tasks.get_range()[0]) return;

                const ParallelDataWeighted gtask = acc_tasks[gid];
                const int row_s = gtask.row_start;
                const int row_e = gtask.row_end;
                const int first_off = gtask.first_row_offset;
                const int last_nnz = gtask.last_row_nnz;

                const std::size_t local_id = item.get_local_id(0);
                const int local_range =
                    static_cast<int>(item.get_local_range(0));
                const int tile_capacity =
                    static_cast<int>(local_capacity);

                for (int r = row_s; r < row_e; ++r) {
                    const int csr_row_begin = acc_row_offsets[r];
                    const int csr_row_end = acc_row_offsets[r + 1];
                    const int deg = csr_row_end - csr_row_begin;
                    if (deg <= 0) continue;

                    int begin = (r == row_s) ? first_off : 0;
                    int end = deg;
                    if (r == row_e - 1 && last_nnz > 0) {
                        end = last_nnz;
                    }
                    begin = sycl::clamp(begin, 0, deg);
                    end = sycl::clamp(end, begin, deg);
                    if (begin >= end) continue;

                    const int chunk_begin = csr_row_begin + begin;
                    const int chunk_end = csr_row_begin + end;
                    const int full_deg = deg;
                    const int out_row = acc_row_origin[r];
                    const bool full_row = begin == 0 && end == full_deg;

                    if (feature_rounds == 2) {
                        // Scheme B (T=2): one sparse traversal, two private
                        // scalar accumulators per work-item.
                        const std::size_t k0 = local_id;
                        const std::size_t k1 = local_id + local_size;
                        const bool valid0 = k0 < K;
                        const bool valid1 = k1 < K;
                        float accum0 = 0.0f;
                        float accum1 = 0.0f;

                        for (int tile_base = chunk_begin;
                             tile_base < chunk_end;
                             tile_base += tile_capacity) {
                            const int remaining = chunk_end - tile_base;
                            const int tile_len =
                                remaining < tile_capacity
                                    ? remaining
                                    : tile_capacity;

                            const int local_t =
                                static_cast<int>(local_id);
                            if (local_t < tile_len) {
                                const int gidx = tile_base + local_t;
                                loc_cols[local_t] = acc_col_index[gidx];
                                loc_vals[local_t] = acc_values[gidx];
                            }
                            item.barrier(sycl::access::fence_space::local_space);

                            for (int t = 0; t < tile_len; ++t) {
                                const int col = loc_cols[t];
                                const float a = loc_vals[t];
                                if (valid0) {
                                    accum0 = sycl::fma(
                                        a, acc_B[col][k0], accum0);
                                }
                                if (valid1) {
                                    accum1 = sycl::fma(
                                        a, acc_B[col][k1], accum1);
                                }
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }

                        if (valid0) {
                            accum0 /= static_cast<float>(full_deg);
                            if (full_row) {
                                acc_C[out_row][k0] = accum0;
                            } else {
                                sycl::atomic_ref<
                                    float,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    atomic_C(acc_C[out_row][k0]);
                                atomic_C.fetch_add(accum0);
                            }
                        }
                        if (valid1) {
                            accum1 /= static_cast<float>(full_deg);
                            if (full_row) {
                                acc_C[out_row][k1] = accum1;
                            } else {
                                sycl::atomic_ref<
                                    float,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    atomic_C(acc_C[out_row][k1]);
                                atomic_C.fetch_add(accum1);
                            }
                        }
                    } else if (feature_rounds == 3) {
                        // Scheme B (T=3): one sparse traversal, three private
                        // scalar accumulators per work-item.
                        const std::size_t k0 = local_id;
                        const std::size_t k1 = local_id + local_size;
                        const std::size_t k2 = local_id + 2 * local_size;
                        const bool valid0 = k0 < K;
                        const bool valid1 = k1 < K;
                        const bool valid2 = k2 < K;
                        float accum0 = 0.0f;
                        float accum1 = 0.0f;
                        float accum2 = 0.0f;

                        for (int tile_base = chunk_begin;
                             tile_base < chunk_end;
                             tile_base += tile_capacity) {
                            const int remaining = chunk_end - tile_base;
                            const int tile_len =
                                remaining < tile_capacity
                                    ? remaining
                                    : tile_capacity;

                            const int local_t =
                                static_cast<int>(local_id);
                            if (local_t < tile_len) {
                                const int gidx = tile_base + local_t;
                                loc_cols[local_t] = acc_col_index[gidx];
                                loc_vals[local_t] = acc_values[gidx];
                            }
                            item.barrier(sycl::access::fence_space::local_space);

                            for (int t = 0; t < tile_len; ++t) {
                                const int col = loc_cols[t];
                                const float a = loc_vals[t];
                                if (valid0) {
                                    accum0 = sycl::fma(
                                        a, acc_B[col][k0], accum0);
                                }
                                if (valid1) {
                                    accum1 = sycl::fma(
                                        a, acc_B[col][k1], accum1);
                                }
                                if (valid2) {
                                    accum2 = sycl::fma(
                                        a, acc_B[col][k2], accum2);
                                }
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }

                        if (valid0) {
                            accum0 /= static_cast<float>(full_deg);
                            if (full_row) {
                                acc_C[out_row][k0] = accum0;
                            } else {
                                sycl::atomic_ref<
                                    float,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    atomic_C(acc_C[out_row][k0]);
                                atomic_C.fetch_add(accum0);
                            }
                        }
                        if (valid1) {
                            accum1 /= static_cast<float>(full_deg);
                            if (full_row) {
                                acc_C[out_row][k1] = accum1;
                            } else {
                                sycl::atomic_ref<
                                    float,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    atomic_C(acc_C[out_row][k1]);
                                atomic_C.fetch_add(accum1);
                            }
                        }
                        if (valid2) {
                            accum2 /= static_cast<float>(full_deg);
                            if (full_row) {
                                acc_C[out_row][k2] = accum2;
                            } else {
                                sycl::atomic_ref<
                                    float,
                                    sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>
                                    atomic_C(acc_C[out_row][k2]);
                                atomic_C.fetch_add(accum2);
                            }
                        }
                    } else {
                        // Scheme A (T>=4): one private accumulator and one
                        // complete sparse traversal per feature round.
                        for (std::size_t feature_round = 0;
                             feature_round < feature_rounds;
                             ++feature_round) {
                            const std::size_t k =
                                local_id + feature_round * local_size;
                            const bool valid = k < K;
                            float accum = 0.0f;

                            for (int tile_base = chunk_begin;
                                 tile_base < chunk_end;
                                 tile_base += tile_capacity) {
                                const int remaining = chunk_end - tile_base;
                                const int tile_len =
                                    remaining < tile_capacity
                                        ? remaining
                                        : tile_capacity;

                                const int local_t =
                                    static_cast<int>(local_id);
                                if (local_t < tile_len) {
                                    const int gidx = tile_base + local_t;
                                    loc_cols[local_t] = acc_col_index[gidx];
                                    loc_vals[local_t] = acc_values[gidx];
                                }
                                item.barrier(
                                    sycl::access::fence_space::local_space);

                                if (valid) {
                                    for (int t = 0; t < tile_len; ++t) {
                                        const int col = loc_cols[t];
                                        const float a = loc_vals[t];
                                        accum = sycl::fma(
                                            a, acc_B[col][k], accum);
                                    }
                                }
                                item.barrier(
                                    sycl::access::fence_space::local_space);
                            }

                            if (valid) {
                                accum /= static_cast<float>(full_deg);
                                if (full_row) {
                                    acc_C[out_row][k] = accum;
                                } else {
                                    sycl::atomic_ref<
                                        float,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,
                                        sycl::access::address_space::global_space>
                                        atomic_C(acc_C[out_row][k]);
                                    atomic_C.fetch_add(accum);
                                }
                            }
                        }
                    }
                }
            });
    });

    q.wait();

    double time_ns = 0.0;
    try {
        time_ns =
            (e.template get_profiling_info<
                 sycl::info::event_profiling::command_end>()
             - e.template get_profiling_info<
                 sycl::info::event_profiling::command_start>());
    } catch (...) {
    }
    time = time_ns / 1e3;

    return C;
}
