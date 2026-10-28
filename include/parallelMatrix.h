#pragma once

#include <vector>

#include <sycl/sycl.hpp>

#include "Matrix.h"
#include "config.h"
#include "utils.h"

// One task per work-group; return mean aggregation in original row order.
// The time output covers the GPU kernel event only, in microseconds.
Defined::RowMajorMatrixXf spmmKernel(
    sycl::queue& q,
    const CSR_Order& adjOrder,
    const Defined::RowMajorMatrixXf& H,
    int dim_hint,
    int local_dimension,
    std::vector<ParallelDataWeighted> tasks,
    double& time);

// Large-K fallback used only when H.cols() exceeds the current device's
// max_work_group_size.  The original CWLB task/row/NNZ mapping and output
// semantics are preserved; work-items can process multiple feature columns.
Defined::RowMajorMatrixXf spmmKernelLargeKFallback(
    sycl::queue& q,
    const CSR_Order& adjOrder,
    const Defined::RowMajorMatrixXf& H,
    int dim_hint,
    int local_dimension,
    std::vector<ParallelDataWeighted> tasks,
    double& time);
