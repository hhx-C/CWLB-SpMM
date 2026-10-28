// Read and validate the graph's two-line CSR representation.
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "data.h"
#include "config.h"

namespace {

bool endsWith(const std::string& s, const std::string& suffix)
{
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

CSR loadTextCSRFile(const std::string& csrName, int& V, int& E)
{
    CSR csr;

    // Dataset paths are relative to the process working directory.
    std::string csrPath = DATA_CSR_FILE_PATH + csrName;

    std::ifstream in(csrPath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open .csr file: " + csrPath);
    }

    std::string line;
    std::string cell;

    if (!std::getline(in, line)) {
        throw std::runtime_error("Failed to read the first line of .csr file: " + csrPath);
    }

    /*---------------- Row offsets ----------------*/
    {
        std::stringstream ss(line);
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                csr.RowOffset.push_back(std::stoi(cell));
            }
        }
    }

    /*---------------- Column indices ----------------*/
    if (!std::getline(in, line)) {
        throw std::runtime_error("Failed to read the second line of .csr file: " + csrPath);
    }

    {
        std::stringstream ss(line);
        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                csr.ColInd.push_back(std::stoi(cell));
            }
        }
    }

    /*---------------- CSR consistency checks ----------------*/
    if (csr.RowOffset.empty()) {
        throw std::runtime_error(".csr file has an empty RowOffset: " + csrPath);
    }

    V = static_cast<int>(csr.RowOffset.size()) - 1;
    E = csr.RowOffset.back();

    if (V < 0) {
        throw std::runtime_error("Invalid RowOffset size. It should be vertex_count + 1");
    }

    if (E < 0) {
        throw std::runtime_error("RowOffset.back(), namely NNZ, is negative");
    }

    if (static_cast<int>(csr.ColInd.size()) != E) {
        throw std::runtime_error(
            "ColInd size does not match RowOffset.back()(NNZ): ColInd=" +
            std::to_string(csr.ColInd.size()) +
            ", NNZ=" + std::to_string(E)
        );
    }

    if (csr.RowOffset.front() != 0) {
        throw std::runtime_error(
            "RowOffset[0] must be 0, current value is " +
            std::to_string(csr.RowOffset.front())
        );
    }

    for (std::size_t i = 1; i < csr.RowOffset.size(); ++i) {
        if (csr.RowOffset[i] < csr.RowOffset[i - 1]) {
            throw std::runtime_error(
                "RowOffset is not monotonically non-decreasing at index " +
                std::to_string(i)
            );
        }

        if (csr.RowOffset[i] > E) {
            throw std::runtime_error(
                "RowOffset contains a value larger than NNZ at index " +
                std::to_string(i)
            );
        }
    }

    for (int c : csr.ColInd) {
        if (c < 0) {
            throw std::runtime_error("ColInd contains a negative column index");
        }
    }

    // Text inputs represent unweighted graphs with unit edge values.
    csr.Value.assign(csr.ColInd.size(), 1.0f);

    return csr;
}
}  // namespace

CSR loadDatasetToCSR(const std::string& datasetName, int& V, int& E)
{
    if (datasetName.empty()) {
        throw std::runtime_error("Dataset name is empty");
    }

    const std::string csrName =
        endsWith(datasetName, ".csr") ? datasetName : datasetName + ".csr";
    return loadTextCSRFile(csrName, V, E);
}
