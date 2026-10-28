// Calibrate the selected GPU and save the four weights for cwlb-spmm.
#include <iostream>
#include <stdexcept>
#include <string>

#include <sycl/sycl.hpp>

#include "config.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    try {

        /*---------------- Calibration parameters ----------------*/
        int feature_count = 128;
        int repeat = 500;
        int work_items = 131072;
        int load_data_mb = 512;  // Load working-set size in MiB.
        int atomic_writers = 8;  // Logical writer groups sharing an output row.

        if (argc != 1 && argc != 6) {
            std::cerr
                << "Usage:\n"
                << "  ./costWrite\n"
                << "  ./costWrite [K] [repeat] [WI] [load_MB] [atomic_writers]\n";
            return 1;
        }

        if (argc == 6) {
            feature_count = std::stoi(argv[1]);
            repeat = std::stoi(argv[2]);
            work_items = std::stoi(argv[3]);
            load_data_mb = std::stoi(argv[4]);
            atomic_writers = std::stoi(argv[5]);
        }

        if (feature_count <= 0 || repeat <= 0 || work_items <= 0 ||
            load_data_mb <= 0 || atomic_writers <= 0) {
            throw std::invalid_argument(
                "All microbenchmark parameters must be positive.");
        }

        /*---------------- Device calibration ----------------*/
        sycl::queue queue(
            sycl::gpu_selector_v,
            sycl::property::queue::enable_profiling{});

        const SpmmCostParams costs = measure_spmm_costs(
            queue,
            feature_count,
            repeat,
            work_items,
            load_data_mb,
            atomic_writers);

        std::cout << "device: "
                  << queue.get_device().get_info<sycl::info::device::name>()
                  << '\n';
        std::cout << "C_compute_ns: " << costs.C_compute_ns << '\n';
        std::cout << "C_load_ns: " << costs.C_load_ns << '\n';
        std::cout << "C_store_ns: " << costs.C_store_ns << '\n';
        std::cout << "C_atomic_ns: " << costs.C_atomic_ns << '\n';

        // Persist the weights in the order expected by the SpMM driver.
        if (!save_costs_one_per_line(costs)) {
            throw std::runtime_error(
                "Failed to write microbenchmark cost file: " + COSTFILEPATH);
        }

        std::cout << "Saved cost parameters to: " << COSTFILEPATH << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
