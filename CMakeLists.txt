cmake_minimum_required(VERSION 3.18)

# Standalone build definition for the CWLB-SpMM production source snapshot.
# This directory intentionally contains only the production main flow and the
# operation-cost calibration program; unrelated experiment targets from the
# project-level CMakeLists.txt are not included here.
project(CWLB_SPMM_SOURCE LANGUAGES NONE)

if(NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "icpx" CACHE FILEPATH
        "C++ compiler for the SYCL targets")
endif()
enable_language(CXX)

find_package(TBB REQUIRED)
find_package(DNNL REQUIRED)

set(CWLB_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/Matrix.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/parallelMatrix.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp"
)

add_executable(
    cwlb-spmm
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    ${CWLB_CORE_SOURCES}
)

add_executable(
    costWrite
    "${CMAKE_CURRENT_SOURCE_DIR}/weightCost.cpp"
    ${CWLB_CORE_SOURCES}
)

set(CWLB_SOURCE_TARGETS
    cwlb-spmm
    costWrite
)

foreach(TARGET_NAME IN LISTS CWLB_SOURCE_TARGETS)
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

    target_compile_options(${TARGET_NAME} PRIVATE
        -fsycl
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend
        --cuda-gpu-arch=sm_89
    )

    target_link_options(${TARGET_NAME} PRIVATE
        -fsycl
        -fsycl-targets=nvptx64-nvidia-cuda
        -Xsycl-target-backend
        --cuda-gpu-arch=sm_89
    )

    target_include_directories(${TARGET_NAME} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        /usr/local/include/eigen3/Eigen
    )

    target_link_libraries(${TARGET_NAME} PRIVATE
        TBB::tbb
        DNNL::dnnl
        OpenCL
        pthread
    )
endforeach()
