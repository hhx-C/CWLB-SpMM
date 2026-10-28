#pragma once

#include <string>
#include <utility>
#include <vector>

#include <sycl/sycl.hpp>

#include "Matrix.h"
#include "config.h"

// Optional GPU bucket ordering; the main CLI uses its host oneDPL sorter.
CSR_Order sortCSRByDegree_SYCL(const CSR& csr, sycl::queue& q);

struct ParallelDataWeighted {
    int row_start;         // First reordered row, inclusive.
    int row_end;           // Last reordered row, exclusive.
    int first_row_offset;  // Nonzero offset within the first row.
    int last_row_nnz;      // End offset in the last row; zero means its full degree.
};

// Device-calibrated relative weights and calibration metadata.
struct SpmmCostParams {
    double C_compute_ns = 0.0;
    double C_load_ns = 0.0;
    double C_store_ns = 0.0;
    double C_atomic_ns = 0.0;

    int K = 0;
    int WI = 0;
    int repeat = 0;
    int atomic_writers_per_addr = 0;
};

// Diagnostic details retained for a failed reference comparison.
struct ErrorElement {
    int result_row = 0;
    int result_col = 0;
    float sycl_value = 0.0f;
    float reference_value = 0.0f;
    float absolute_error = 0.0f;
    std::vector<std::pair<int, float>> csr_row_elements;
    std::vector<float> dense_col_elements;
};

// Calibrate compute, load, direct-store, and atomic-update weights separately.
SpmmCostParams measure_spmm_costs(
    sycl::queue& q,
    int K = 128,
    int repeat = 500,
    int WI = 131072,
    int data_MB_for_load = 512,
    int atomic_target_writers = 8,
    const std::vector<int>& atomic_group_candidates = {});

// Pack complete rows and split oversized rows under the weighted capacity.
std::vector<ParallelDataWeighted> costWeightedTaskPartitioning(
    const CSR_Order& csr,
    const SpmmCostParams& costs,
    double capacity_weight);

// Cost-file order: compute, load, store, atomic; one coefficient per line.
bool save_costs_one_per_line(
    SpmmCostParams costs,
    const std::string& filepath = COSTFILEPATH,
    bool append = false,
    int precision = 6);

bool read_costs_one_per_line(
    SpmmCostParams& out,
    const std::string& filepath = COSTFILEPATH);

std::vector<std::pair<int, float>> get_csr_row_elements(
    const CSR& csr,
    int row);

// Check a caller-provided reference using an absolute error tolerance.
bool validate_custom_csr_result(
    const Defined::RowMajorMatrixXf& sycl_result,
    const Defined::RowMajorMatrixXf& reference,
    const CSR& csr,
    const Defined::RowMajorMatrixXf& dense_matrix,
    float abs_tolerance = 1e-4f);
