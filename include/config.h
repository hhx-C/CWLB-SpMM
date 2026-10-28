#pragma once
// Shared timing settings, runtime paths, and dense-matrix layout.
#include <string>
#include <vector>
#include <Eigen>
using namespace std;

// Timed samples = LOOPCOUNT - WARMUPCOUNT; LOOPCOUNT includes warm-up.
inline const int WARMUPCOUNT = 20;
inline const int LOOPCOUNT = 70;
// Resolve these paths from the project's build/ working directory.
inline const std::string& COSTFILEPATH = "../resources/spmm_costs.txt";

constexpr const char* DATA_CSR_FILE_PATH
        = "../resources/datasets/";

namespace Defined {
    // Adjacent feature columns are contiguous in memory.
    using RowMajorMatrixXf = Eigen::Matrix<
        float, 
        Eigen::Dynamic, 
        Eigen::Dynamic, 
        Eigen::RowMajor
    >;
}
