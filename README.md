# CWLB-SpMM

This repository provides the standalone SYCL implementation of **CWLB-SpMM:
Cost-Weighted Load Balancing for SYCL-Based SpMM Kernels in Graph Neural
Networks**. CWLB-SpMM combines device-calibrated operation costs, degree-based
row reordering, and cost-weighted task partitioning to balance sparse
aggregation across GPU work-groups.

The repository builds two executables: `cwlb-spmm` for graph aggregation and
segment-size evaluation, and `costWrite` for operation-cost calibration. The
driver performs FP32 mean aggregation with a reproducible, internally generated
dense feature matrix; it is not a complete GNN training framework.

## Citation

Please cite our paper if you use this code in your research.

<!-- Add the BibTeX citation below. -->
```bibtex

```

## Requirements

- Linux and an NVIDIA GPU. The development GPU is an RTX 4080 SUPER.
- Intel oneAPI DPC++/C++ compiler (`icpx`) with a working NVIDIA CUDA backend.
  The source has been built with Intel oneAPI 2025.0.
- A CUDA toolkit and NVIDIA driver compatible with the installed SYCL backend.
- CMake 3.18 or newer and C++17 support.
- oneDPL headers, oneTBB, oneDNN, and the OpenCL loader. The host oneDPL
  preprocessing uses the parallel execution policy backed by oneTBB.
- Eigen 3 headers. The current include path is
  `/usr/local/include/eigen3/Eigen`; adjust `CMakeLists.txt` if Eigen is
  installed elsewhere.

The supplied build configuration targets the CUDA backend and `sm_89`.
For another GPU architecture, update the architecture flag in both the compile
and link options. A different backend also requires corresponding build and
runtime configuration; the supplied CMake file is NVIDIA-specific.

## Download datasets

First clone the repository:

```bash
git clone https://github.com/hhx-C/CWLB-SpMM.git
cd CWLB-SpMM
```

Download link: <!-- Add the dataset download URL here. -->

Download the dataset archive separately and place it in the repository root.
The download URL is reserved above. The archive filenames below are examples;
use the commands matching the actual archive format and internal layout.

### Extract the archive

For a `.tar.gz` archive containing the `.csr` files directly at its root:

```bash
mkdir -p resources/datasets
tar -tzf datasets.tar.gz
tar --keep-old-files -xzf datasets.tar.gz -C resources/datasets
```

If the listing instead shows exactly one enclosing directory, such as
`datasets/cora.csr`, use the following extraction command **instead**:

```bash
tar --keep-old-files -xzf datasets.tar.gz --strip-components=1 -C resources/datasets
```

For a `.zip` archive containing the `.csr` files directly at its root:

```bash
mkdir -p resources/datasets
unzip -l datasets.zip
unzip -n datasets.zip -d resources/datasets
```

These commands preserve existing files. If an archive contains additional
directory levels, arrange the extracted files to match the layout below.
Do not leave an extra `datasets/` or `csr/` layer inside
`resources/datasets/`.

### Dataset location and format

Place all 19 `.csr` files **directly in `resources/datasets/` at the
repository root**:

```text
CWLB-SpMM/
  CMakeLists.txt
  README.md
  main.cpp
  weightCost.cpp
  include/
  src/
  resources/
    datasets/
      cora.csr
      oregon-1.csr
      ...
      reddit.dgl.csr
      products.csr
  build/
```

Only the two-line text CSR format is accepted:

1. First line: comma-separated, zero-based row offsets.
2. Second line: comma-separated, zero-based column indices.

Edge values are implicitly set to one. There is no matrix-size header or
separate value line. A binary CSR file or a Matrix Market file cannot be used
simply by renaming its extension.

For example, a three-row graph with four nonzeros is represented as:

```text
0,2,3,4
1,2,0,1
```

The driver accepts either `cora` or `cora.csr`. For the graph called
`reddit` in the paper, use the filename `reddit.dgl.csr` and the command-line
name `reddit.dgl` (or `reddit.dgl.csr`).

### The 19 benchmark graphs

The statistics below follow the dataset table in the paper. NNZ denotes the
number of stored adjacency-matrix nonzeros.

| Dataset | Nodes | NNZ | Average degree | Maximum degree |
| --- | ---: | ---: | ---: | ---: |
| cora | 2,708 | 10,556 | 3.89 | 168 |
| oregon-1 | 11,492 | 46,818 | 4.07 | 2,389 |
| pubmed | 19,717 | 99,203 | 5.03 | 171 |
| wiki-Vote | 8,297 | 103,689 | 12.49 | 893 |
| as-caida | 31,379 | 106,762 | 3.40 | 2,628 |
| email-Enron | 36,692 | 367,662 | 10.02 | 1,383 |
| email-EuAll | 265,214 | 420,045 | 1.58 | 930 |
| soc-Slashdot0811 | 77,360 | 905,468 | 11.70 | 2,508 |
| arxiv | 169,343 | 1,166,243 | 6.88 | 13,155 |
| artist | 50,515 | 1,638,396 | 32.43 | 1,469 |
| collab | 235,868 | 2,358,104 | 9.99 | 671 |
| amazon0601 | 403,394 | 5,478,357 | 13.58 | 2,547 |
| am | 881,680 | 5,668,682 | 6.42 | 154,828 |
| yelp | 716,847 | 13,954,819 | 19.46 | 4,886 |
| wikikg2 | 2,500,604 | 16,109,182 | 6.44 | 911 |
| citation | 2,927,963 | 30,387,995 | 10.37 | 1,738 |
| ppa | 576,289 | 42,463,862 | 73.68 | 3,241 |
| reddit | 232,965 | 114,615,891 | 491.98 | 21,657 |
| products | 2,449,029 | 123,718,280 | 50.51 | 17,481 |

## Build

From the repository root, initialize the compiler environment and build:

```bash
source /opt/intel/oneapi/2025.0/oneapi-vars.sh
cmake -S . -B build
cmake --build build -j2
cd build
export ONEAPI_DEVICE_SELECTOR=cuda:gpu
```

Adjust the oneAPI initialization path for your installation. Both executables
are generated in `build/`. All commands below must run with `build/` as the
working directory: dataset, coefficient, and output paths are relative to that
directory, not to the executable's location.

## Run

**You must run the operation-cost microbenchmarks before using CWLB-SpMM.**
The algorithm requires their calibrated weights for task partitioning;
`cwlb-spmm` does not generate these weights automatically.
The examples use `cora` and feature dimension `K=128`; replace them with
another dataset and dimension as needed. The paper evaluates
`K = 32, 64, 96, 128`.

### Required first step: run the microbenchmarks

From `build/`, run calibration on the selected GPU:

```bash
./costWrite
```

This generates `../resources/spmm_costs.txt`, containing the four operation-cost
weights required by **every** SpMM execution mode below, including candidate
generation and candidate search. If this file is missing or cannot be read,
`cwlb-spmm` exits with an error. Calibration is therefore a mandatory setup
step, not an optional performance experiment.

Once generated, the weights can be reused across datasets and `S` values on
the same GPU and build configuration; calibration is not required before
every invocation. Recalibrate when changing the target GPU or relevant
compiler/build configuration. Running calibration again replaces the existing
coefficient file; its parent directory is created automatically if needed.

The equivalent explicit default configuration is:

```bash
./costWrite 128 500 131072 512 8
```

The five positional parameters are:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| K | 128 | Microbenchmark feature dimension |
| repeat | 500 | Timed launches per operation benchmark |
| WI | 131,072 | Work-item count |
| load_MB | 512 | Load working-set size in MiB |
| atomic_writers | 8 | Logical writer groups mapped to the same output row |

Calibration uses its own 20 warm-up launches per operation benchmark,
independently of the SpMM driver's repetition settings.

### 1. Evaluate the three recommended candidates: `--candidates`

```bash
./cwlb-spmm cora 128 --candidates
```

The program selects three empirical wave parameters `w` from the graph's NNZ
range and computes each candidate as:

```text
S = max(1, ceil(NNZ / (device_compute_units * w)))
```

It evaluates all three segment sizes and prints each candidate's ID, `w`,
`S`, and mean SpMM kernel time, followed by the fastest measured candidate.
The compute-unit count is queried from the selected GPU. This mode prints
results without appending to `results.txt`.

### 2. Search near the best candidate: `--candidate-search`

```bash
./cwlb-spmm cora 128 --candidate-search
```

The program first evaluates the same three candidates. For ordered candidate
sizes `S1 <= S2 <= S3`, it selects an inclusive search interval:

| Fastest initial candidate | Search interval |
| --- | --- |
| S1 | `[max(1, 2*S1-S2), S2]` |
| S2 | `[S1, S3]` |
| S3 | `[S2, 2*S3-S2]`, capped at the int32 limit |

By default, it generates six near-uniform integer search points, including both
endpoints when possible. The interval and planned `S` values are printed
before a `yes/no` prompt.

- Enter `yes` to evaluate the planned points and append both the initial and
  search-phase `S`/time pairs to `../resources/results.txt`.
- Enter `no`, or provide end-of-input, to exit after the initial candidate
  evaluation without creating or changing `results.txt`.

The final best `S` is selected from both phases. This is a bounded empirical
search, not a guarantee of the global optimum.

### 3. Set the number of search points

```bash
./cwlb-spmm cora 128 --candidate-search 10
```

Request ten neighborhood search points after the three initial candidates.
The interval rule and confirmation prompt are unchanged. A narrow interval
may contain fewer distinct integer values; a requested count of one selects
the interval midpoint. The count refers to search points, not kernel timing
repetitions.

### 4. Set a manual search step

```bash
./cwlb-spmm cora 128 --candidate-search --step 3
```

Use an integer step of three within the selected interval instead of a point
count. The upper endpoint is also included if the step does not land on it.
The interval rule and confirmation prompt are unchanged. Search count and
step must be positive, and plans exceeding 10,000 actual points are rejected.

### 5. Run a fixed segment size: `--S`

```bash
./cwlb-spmm cora 128 --S 8
```

Run one configuration with `S=8`. No candidate generation or neighborhood
search is performed.

### 6. Run with a wave parameter

```bash
./cwlb-spmm cora 128 64
```

Here the final positional argument is `w=64`, **not `S=64`**.
The program derives `S` using the NNZ/compute-unit formula above and evaluates
that single configuration.

### 7. Sweep an explicit range of segment sizes

```bash
./cwlb-spmm cora 128 2 30 2
```

The last three arguments are `S_begin`, `S_max`, and `S_step`.
This example tests `S = 2, 4, ..., 30`, appends `S`/time pairs to
`../resources/results.txt`, and prints the fastest tested `S`.
Unlike candidate search, an explicit sweep starts without a confirmation
prompt. In general, `S_max` is included only if the step reaches it exactly.

## Timing and output

SpMM repetition settings are defined in `include/config.h`. Currently,
`WARMUPCOUNT=20` and `LOOPCOUNT=70`: the latter is the **total** launch count,
so the reported mean uses 50 timed launches after 20 warm-ups.
These are the current source settings, not a replacement for the historical
timing protocol used in the paper.

Times printed as `SpMM time` are arithmetic-mean SYCL kernel-event times in
microseconds. They exclude dataset loading, calibration, preprocessing,
allocation, host/device transfers, and file I/O.

| Location relative to the repository root | Contents |
| --- | --- |
| `resources/datasets/` | The separately downloaded CSR files |
| `resources/spmm_costs.txt` | Four calibration coefficients: compute, load, store, atomic |
| `resources/results.txt` | Appended explicit-sweep or confirmed candidate-search results |

Ordinary fixed-`S`, wave, and candidate-only runs print their results to the
terminal. For example, to retain a candidate-only log:

```bash
./cwlb-spmm cora 128 --candidates > ../resources/cora_k128_candidates.log
```

## Common setup issues

- **Cannot open a CSR file:** check the exact filename and ensure the working
  directory is `build/`; `resources/datasets/` must not contain an extra
  enclosing folder.
- **Cannot read the cost file:** run `./costWrite` from `build/` first.
- **CMake cannot find TBB or DNNL:** initialize the dependency environment and,
  if needed, set `CMAKE_PREFIX_PATH` to their installation prefixes.
- **CUDA backend or architecture errors:** check that `icpx` has NVIDIA
  backend support and that both architecture flags match the target GPU.
