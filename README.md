# ParAMD: Parallel Approximate Minimum Degree Algorithm

[![License](https://img.shields.io/badge/License-BSD_3--Clause-g.svg)](https://opensource.org/licenses/BSD-3-Clause) [![Version](https://img.shields.io/badge/Version-v1.0.0-blue.svg)]() [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.15283182.svg)](http://dx.doi.org/10.5281/zenodo.15283182)

[[Installation]](#installation) [[Using ParAMD]](#using-paramd) [[Replicating Our benchmarks]](#replicating-our-benchmarks) [[Citation and Acknowledgments]](#citation-and-acknowledgments) [[Copyright]](#copyright)

ParAMD is a shared memory parallel implementation of the approximate minimum degree (AMD) algorithm with multiple elimination via distance-2 independent sets.

For more information, please read our preprint paper [Parallelizing the Approximate Minimum Degree Ordering Algorithm: Strategies and Evaluation](https://arxiv.org/abs/2504.17097) on arXiv.

## Installation
### Build
To build the shared library `libparamd.so` and the binary `paramd`, run the following commands from the root of this repository:

```sh
mkdir build && cd build
cmake ..
cmake --build .
```

This will place the output files in the `build/` directory.

### Install
To install the header file `paramd.h`, the shared library `libparamd.so`, and the binary `paramd` to a custom location, run:

```sh
cmake --install . --prefix /path/to/install/
```

Replace `/path/to/install/` with your desired installation directory.

### Optional: Enable SuiteSparse AMD
If you have [SuiteSparse AMD](https://github.com/DrTimothyAldenDavis/SuiteSparse/tree/dev/AMD) installed, you can enable support for it by specifying its path during the configuration step. Run the following instead of `cmake ..`:

```sh
cmake .. -DSUITESPARSE_ROOT=/path/to/amd/
```
Make sure the following files exist:

Header: `/path/to/amd/include/suitesparse/amd.h`

Library: `/path/to/amd/lib64/libamd.so`, `/path/to/amd/lib64/libamd.a`, `/path/to/amd/lib/libamd.so`, or `/path/to/amd/lib/libamd.a`

## Using ParAMD
### Command-Line Interface
The easiest way to use ParAMD is via the command-line binary `paramd`. To view all available options, run:

```sh
./paramd -h # in the build/ directory, or paramd -h if installed
```

You will see the following usage message:

```
Usage: paramd [OPTIONS]
Options:
  -h, --help                               Show this help message
  --matrix <path>                          Path to the input matrix in mtx format (required)
  --algo <paramd, amd (if available)>      Algorithm to benchmark (default: paramd) 
  --mult <double>                          Multiplicative relaxation factor (default: 1.1)
  --lim <double>                           Limitation factor (default: 8192), will be divided by #threads
  --mem <double>                           Extra memory factor (default: 1.5)
  --seed <64-bit integer>                  Seed used to randomly permute the matrix (default: 1)
  --breakdown                              Enable time breakdown (default: false)
  --stat                                   Enable statistics (default: false)
  --sym                                    Specify sym if the matrix is known to be symmetric (default: false)
```
#### Argument Descriptions and Outputs
The argument descriptions are as follows:
* `-h, --help`: Display the help message and exit.
* `--matrix <path>`: Required. Specify the path to the input matrix file in [Matrix Market](https://math.nist.gov/MatrixMarket/formats.html) (.mtx) format.
* `--algo <paramd, amd>`: Select the ordering algorithm. (Default: `paramd`)
  * `paramd`: Use ParAMD.
  * `amd`: Use SuiteSparse AMD if available.
* `--mult <double>`: The multiplicative relaxation factor to the approximate minimum degree used when selecting pivots. Only apply to ParAMD. (Default: `1.1`)
* `--lim <double>`: The limitation factor used when selecting pivots, will be divided by the number of threads. Only apply to ParAMD. (Default: `8192`)
* `--mem <double>`: The extra memory factor for the graph structure. Only apply to ParAMD. (Default: `1.5`)
* `--seed <64-bit integer>`: The seed used to randomly permute the matrix in advance to decouple tie-breaking issues. Setting this to `0` means no permutation is used beforehand. (Default: `1`) 
* `--breakdown`: Enable time breakdown. Only apply to ParAMD. (Default: `false`)
* `--stat`: Enable statistics for the distribution of the sizes of distance-2 independent sets. Only apply to ParAMD. (Default: `false`)
* `--sym`: Specify this if the matrix is known to be symmetric. Only apply to ParAMD. (Default: `false`)

The fill-reduction ordering will be stored as a permutation at `<path>.perm` where `<path>` is the path to the input matrix.
#### Example Usage

```sh
./paramd --matrix ../examples/test.mtx --algo paramd \
         --mult 1.1 --lim 8192 --mem 1.5 --seed 1 \
         --breakdown --stat --sym # running from the build/ directory
```

#### Performance Tip
We recommend configuring your environment for OpenMP based on your hardware by setting the environment variables, including but not limited to `OMP_NUM_THREADS` and `OMP_PLACES`.

### Integration
ParAMD provides a header file and a shared library to allow easy integration into custom workflows. Once installed and your environment is properly configured, you can include ParAMD in your project with:

```cpp
#include "paramd/paramd.h"
```

And link against the shared library `libparamd.so`.

#### API Overview
* `paramd::config config()`:
  A configuration structure constructing `config` with default values corresponding to the command-line options. You can modify its fields as needed. Use the member function `print()` to display the current configuration.
* `paramd::matrix A(std::string fp)`:
  The matrix structure with a constructor constructing a sparse matrix `A` by reading a Matrix Market (.mtx) file specified by the file path `fp`.
* `uint64_t paramd::paramd(const vtype n, const vtype *rowptr, const etype *colidx, vtype *perm, const config &config)`:
  Run the ParAMD algorithm with:
  * `n`: the number of rows/columns in the matrix.
  * `rowptr`, `colidx`: the CSR/CSC representation (will be symmetrized) of matrix `A`. `A.rowptr.data()` and `A.colidx.data()` can be used if the matrix structure is constructed. The length of `rowptr` is `n+1` and the length of `colidx` is `rowptr[n]`.
  * `perm`: an output array of length `n` that will contain the computed fill-reducing permutation.
  * `config`: the configuration options.
  
  The function returns the number of fill-ins in the Cholesky factorization (excluding the diagonal). Datatypes `vtype` and `etype` are both `int32_t`.

For practical usage examples, helper functions for randomly permuting the matrix and reverting for the resulting permutation, and the full integration workflow, refer to [bin/main.cpp](bin/main.cpp).

A minimal example is shown below:
```cpp
#include <iostream>
#include "paramd/paramd.h"
#include <vector>

int main(int argc, char* argv[]) {
  paramd::config config;
  paramd::matrix A("examples/test.mtx");
  std::vector<paramd::vtype> perm(A.n);
  uint64_t fill = paramd::paramd(A.n, A.rowptr.data(), A.colidx.data(), perm.data(), config);
  std::cout << "#Fill = " << fill << std::endl;
  paramd::write_perm("examples/test.mtx.perm", perm);
  return 0;
}
```

## Replicating Our benchmarks

### Hardware Requirements
Our benchmarks require a CPU with at least 64 physical cores, 256 GB of RAM, and 128 GB of available disk space.

### Software Requirements
* `CMake >= v3.22`
* `Python >= v3.13`, with the following packages:
  * `Matplotlib >= v3.10.1`
  * `NumPy >= v2.2.5`
  * `pandas >= v2.2.3`
  * `seaborn >= v0.13.2`

We recommend setting up the environment using [conda](https://docs.conda.io/projects/conda/en/latest/index.html) with the provided [environment.yml](environment.yml) configuration file. If conda is not already installed on your system, please refer to its [official documentation](https://docs.conda.io/projects/conda/en/latest/index.html) for installation instructions.

Additionally, ensure that you have GCC installed. It must supports `#pragma omp atomic compare`. In our experiments, we used version `v13.2.1`.

To create and activate the environment, run:
```sh
conda env create -f environment.yml
conda activate paramd-benchmarks
```

### Running All Benchmarks

We provide a single script, `all.sh`, that runs all benchmark stages including `download.sh`, `build.sh`, `scaling.sh`, `breakdown.sh`, `distribution.sh`, and `tuning.sh` reproducing key tables and figures. The entire workflow is expected to take 4 hours.

To run all benchmarks, navigate to the root of this repository and execute the following command:

```sh
./scripts/all.sh /path/to/a/clean/scratch/dir/
```

The `/path/to/a/clean/scratch/dir/` directory will be used to store the dataset, benchmark results, and intermediate files.

Once the benchmarks are complete, the following output files will be generated:
* `/path/to/a/clean/scratch/dir/plots/scaling.csv`: An end-to-end comparison between SuiteSparse AMD and our ParAMD, using 1 thread (1t) and 64 threads (64t). Matrices are randomly permuted five times, with all methods evaluated on the same set of permutations to decouple tie-breaking issues. We report the median ordering time and fill-in across the five runs. Speedups and fill-in ratios are calculated based on the medians.

* `/path/to/a/clean/scratch/dir/plots/scaling.html`: The same data as in the CSV file, but formatted as an HTML table for easier viewing.

* `/path/to/a/clean/scratch/dir/plots/breakdown.png`: A runtime breakdown of our ParAMD as the number of threads scales from 1 to 64.

* `/path/to/a/clean/scratch/dir/plots/distribution.png`: Violin plots showing the distribution of the sizes of distance-2 independent sets across elimination steps for ParAMD. The plots are cropped to reflect the actual range of the data. The dotted line marks the threshold of 64, representing the minimum size needed to fully utilize all 64 threads.

* `/path/to/a/clean/scratch/dir/plots/tuning.png`:  The impact of the relaxation factor $mult$ and the limitation factor $lim$ on core AMD runtime, distance-2 independent set selection time, and ordering quality for our ParAMD. Results are shown for matrices `nd24k` and `nlpkkt240`, representing the worst and best scalability cases, respectively. This experiment is conducted using 64 threads. Lighter colors indicate better performance.

Exemplar outputs executed on an AMD EPYC 7763 CPU are available in the [plots](plots/) directory.

For completeness, a detailed description of each script is provided below.

#### Downloading Dataset
To download the dataset, execute the following command from the root of this repository:

```sh
./scripts/download.sh /path/to/a/clean/scratch/dir/
```

This will download the benchmark matrices to `/path/to/a/clean/scratch/dir/dataset/`. This takes about 30 minutes, which may vary depending on internet bandwidth. The matrices used in our benchmarks are from [SuiteSparse Matrix Collection](https://sparse.tamu.edu/), with details provided below, sorted by the number of nonzeros:

| Matrix Name | #rows | #nonzeros | Symmetric | Problem Type |
| :- | -: | -: | :-: | :- |
| [pre2](https://sparse.tamu.edu/ATandT/pre2) | 659K | 5.83M | No | Circuit Simulation |
| [nd24k](https://sparse.tamu.edu/ND/nd24k) | 72.0K | 28.7M | Yes | 3D Mesh |
| [ldoor](https://sparse.tamu.edu/GHS_psdef/ldoor) | 952K | 42.5M | Yes | Structural |
| [dielFilterV3real](https://sparse.tamu.edu/Dziekonski/dielFilterV3real) | 1.10M | 89.3M | Yes | Electromagnetics |
| [Flan_1565](https://sparse.tamu.edu/Janna/Flan_1565) | 1.56M | 114M | Yes | Structural |
| [HV15R](https://sparse.tamu.edu/Fluorem/HV15R) | 2.02M | 283M | No | Computational Fluid Dynamics |
| [Queen_4147](https://sparse.tamu.edu/Janna/Queen_4147) | 4.15M | 317M | Yes | Structural |
| [stokes](https://sparse.tamu.edu/VLSI/stokes) | 11.4M | 349M | No | Semiconductor Process |
| [nlpkkt240](https://sparse.tamu.edu/Schenk/nlpkkt240) | 28.0M | 761M | Yes | Optimization |

#### Building SuiteSparse AMD and ParAMD
To build SuiteSparse AMD and ParAMD, run the following command from the root of this repository:

```sh
./scripts/build.sh /path/to/a/clean/scratch/dir/
```

This will first clone the [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) repository, build AMD, and install the shared library to `/path/to/a/clean/scratch/dir/`. Afterward, ParAMD will be built with SuiteSparse AMD enabled. The estimated compilation time is 2 minutes.

#### Running Scaling Experiment and End-to-End Comparison
To run the scaling experiment and end-to-end comparison, execute the following command from the root of this repository:

```sh
./scripts/scaling.sh /path/to/a/clean/scratch/dir/
```

This will generate two tables: `/path/to/a/clean/scratch/dir/plots/scaling.csv` and `/path/to/a/clean/scratch/dir/plots/scaling.html`. Both tables contain the same data, with the latter formatted for better human readability. The experiment performs an end-to-end comparison between SuiteSparse AMD and our ParAMD, using 1 thread (1t) and 64 threads (64t). Matrices are randomly permuted five times, and all methods are evaluated on the same set of permutations to eliminate tie-breaking issues. We report the median ordering time and fill-in across the five runs, with speedups and fill-in ratios calculated based on the median values. This takes about 2 hours.

#### Runtime Breakdown Experiment
To run the time breakdown experiment, execute the following command from the root of this repository:

```sh
./scripts/breakdown.sh /path/to/a/clean/scratch/dir/
```

This will generate a figure at `/path/to/a/clean/scratch/dir/plots/breakdown.png`. The experiment provides a runtime breakdown of our ParAMD as the number of threads scales from 1 to 64. This takes about 40 minutes.

#### Running Distribution Experiment
To run the distribution experiment, execute the following command from the root of this repository:

```sh
./scripts/distribution.sh /path/to/a/clean/scratch/dir/
```

This will generate a figure at `/path/to/a/clean/scratch/dir/plots/distribution.png`. The figure displays violin plots showing the distribution of the sizes of distance-2 independent sets across elimination steps in ParAMD. The plots are cropped to show the actual range of the data, with a dotted line marking the threshold of 64, which represents the minimum size required to fully utilize all 64 threads. This takes about 2 minutes.

#### Running Tuning Experiment
To run the tuning experiment, execute the following command from the root of this repository:

```sh
./scripts/tuning.sh /path/to/a/clean/scratch/dir/
```

This will generate a figure at `/path/to/a/clean/scratch/dir/plots/tuning.png`. The figure illustrates the impact of the relaxation factor $mult$ and the limitation factor $lim$ on core AMD runtime, distance-2 independent set selection time, and ordering quality for ParAMD. Results are shown for matrices `nd24k` and `nlpkkt240`, representing the worst and best scalability cases, respectively. This experiment is conducted using 64 threads. Lighter colors indicate better performance. This takes about 40 minutes.



## Citation and Acknowledgments
If you find this code useful, please cite our paper:

```bibtex
@misc{paramd,
      title={Parallelizing the Approximate Minimum Degree Ordering Algorithm: Strategies and Evaluation}, 
      author={Yen-Hsiang Chang and Aydın Buluç and James Demmel},
      year={2025},
      eprint={2504.17097},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2504.17097}, 
}
```

This software optionally uses AMD (Approximate Minimum Degree), primarily used when benchmarking, from the [SuiteSparse project](https://github.com/DrTimothyAldenDavis/SuiteSparse) which is licensed under the BSD 3-Clause License. Our codebase extracts the workflow from SuiteSparse AMD, which is a sequential implementation, where we enable parallelism via multiple elimination using distance-2 independent sets to build our ParAMD.

## Copyright

Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved. 

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.