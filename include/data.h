#pragma once

#include <string>

#include "Matrix.h"

// Load two-line text CSR; datasetName may include the .csr extension.
CSR loadDatasetToCSR(const std::string& datasetName, int& V, int& E);
