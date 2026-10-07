# Transformer Self-Attention Performance Study

A reproducible implementation and performance study of Transformer self-attention using sequential CPU, OpenMP, and CUDA implementations.

This project focuses on understanding where self-attention spends computation, how parallel execution changes performance, and how targeted CUDA optimizations affect end-to-end latency.

The reported timings are hardware- and environment-dependent. The values included in this repository are reference measurements from the stated environment and should be interpreted as comparative benchmarks rather than universal performance guarantees.

## Overview

The project investigates self-attention from a systems and performance perspective.

### Objectives

- Implement a sequential CPU reference.
- Implement an OpenMP parallel CPU version.
- Implement a CUDA GPU version.
- Validate parallel implementations against the CPU reference.
- Measure execution time as sequence length increases.
- Profile the individual attention stages.
- Identify performance bottlenecks.
- Test targeted CUDA optimizations.
- Record both successful and unsuccessful optimization attempts.
- Make the experiments reproducible on Linux, Google Colab, and other CUDA-capable environments.

The current study intentionally uses a simple implementation so that the effect of individual optimizations can be isolated and understood.

## Mathematical Formulation

For an input matrix

`X ∈ R^(N × D)`,

the query, key, and value matrices are

- `Q = XW_Q`
- `K = XW_K`
- `V = XW_V`

Scaled dot-product self-attention is

`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`

The implementation consists of these main stages:

1. QKV projection
2. Transpose of `K`
3. `QK^T`
4. Scaling by `1 / sqrt(d_k)`
5. Row-wise softmax
6. Attention-weighted `V` (AV)

The major matrix-multiplication stages have approximately `O(N^2 d)` computational complexity, while row-wise softmax has approximately `O(N^2)` work.

## Project Structure

```text
attension-project/
├── README.md
├── benchmarks/
│   ├── cpu_attention.cpp
│   └── openmp_attention.cpp
├── cuda/
│   ├── hello_cuda.cu
│   ├── vector_add.cu
│   ├── self_attention.cu
│   ├── self_attention_e2e.cu
│   ├── self_attention_profile.cu
│   ├── self_attention_tiled.cu
│   └── self_attention_softmax.cu
├── include/
│   ├── attention.hpp
│   ├── attention_openmp.hpp
│   ├── matrix.hpp
│   └── matrix_ops.hpp
├── results/
│   ├── cpu/
│   ├── openmp/
│   └── cuda/
├── src/
│   ├── attention.cpp
│   ├── attention_openmp.cpp
│   ├── matrix.cpp
│   └── matrix_ops.cpp
└── tests/
    ├── test_attention.cpp
    ├── test_attention_openmp.cpp
    ├── test_matmul.cpp
    ├── test_matrix.cpp
    └── test_softmax.cpp
```

Compiled executables and object files are intentionally not required in the repository. They should be generated locally using the supplied source code.

## Software and Hardware

### CPU development environment

The CPU experiments were performed in Google Colab on an Intel Xeon environment exposed as:

- CPU(s): 2
- Model name: Intel(R) Xeon(R) CPU @ 2.00GHz
- Thread(s)/core: 2
- Core(s)/socket: 1

Therefore, the OpenMP experiment should not be interpreted as scaling across a large multicore CPU. The Colab runtime exposed one physical core with two logical threads.

### GPU reference environment

- GPU: NVIDIA Tesla T4
- VRAM: 15360 MiB
- Driver: 580.82.07
- CUDA: 13.0
- Precision: FP32

### Compiler

- C++17
- NVIDIA `nvcc`

## Building and Running the Project

### 1. Sequential CPU implementation

From the project root:

```bash
g++ -std=c++17 -O2 \
    -Iinclude \
    src/matrix.cpp \
    src/matrix_ops.cpp \
    src/attention.cpp \
    benchmarks/cpu_attention.cpp \
    -o benchmarks/cpu_attention
```

Run:

```bash
./benchmarks/cpu_attention
```

### 2. OpenMP implementation

Compile:

```bash
g++ -std=c++17 -O2 -fopenmp \
    -Iinclude \
    src/matrix.cpp \
    src/matrix_ops.cpp \
    src/attention_openmp.cpp \
    benchmarks/openmp_attention.cpp \
    -o benchmarks/openmp_attention
```

Run:

```bash
./benchmarks/openmp_attention
```

The OpenMP implementation parallelizes matrix multiplication and row-wise operations while retaining the sequential implementation as the correctness reference.

### 3. CUDA implementations

#### Baseline CUDA

```bash
nvcc -O2 -std=c++17 \
    cuda/self_attention.cu \
    -o cuda/self_attention
```

#### End-to-end CUDA benchmark

```bash
nvcc -O2 -std=c++17 \
    cuda/self_attention_e2e.cu \
    -o cuda/self_attention_e2e
```

#### Stage profiler

```bash
nvcc -O2 -std=c++17 \
    cuda/self_attention_profile.cu \
    -o cuda/self_attention_profile
```

#### Tiled GEMM experiment

```bash
nvcc -O2 -std=c++17 \
    cuda/self_attention_tiled.cu \
    -o cuda/self_attention_tiled
```

#### Parallel-softmax experiment

```bash
nvcc -O2 -std=c++17 \
    cuda/self_attention_softmax.cu \
    -o cuda/self_attention_softmax
```

## Correctness Validation

The project contains tests for:

- Matrix construction and indexing
- Matrix multiplication
- Matrix transpose
- Numerically stable softmax
- Scaled dot-product attention
- Complete self-attention
- OpenMP attention

The GPU implementations compare their output against the CPU reference.

For the T4 experiments, the maximum absolute error remained below `1e-4`.

Authoritative T4 baseline maximum errors:

| N | Max absolute error |
|---:|------------------:|
| 64 | `4.6492e-05` |
| 128 | `7.0095e-05` |
| 256 | `7.4744e-05` |
| 512 | `8.7738e-05` |
| 1024 | `9.3460e-05` |

Small differences are expected because CPU and GPU floating-point operations can accumulate values in different orders.

## Numerical Stability

Softmax is implemented using the numerically stable form:

`softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))`

Subtracting the maximum value prevents large positive values from causing exponential overflow.

The implementation was tested with large input values around `1000`.

## Experimental Configuration

Unless otherwise stated:

- `D = 128`
- `d_k = 128`
- `d_v = 128`
- `N = 64, 128, 256, 512, 1024`
- `Repetitions = 5`
- `Datatype = float32`

The benchmark uses deterministic input generation so that different implementations can be compared consistently.

### Experimental Results Sources

- CPU and OpenMP reference measurements are stored under `results/`.
- T4 baseline and tiled-GEMM measurements are stored under `results/cuda/`.
- The latest CUDA end-to-end, stage-profiler, and parallel-softmax reference measurements reported in this README were recorded from the corresponding T4 Google Colab benchmark runs.

## Results

### Sequential CPU results

| N | QKV (ms) | QK^T (ms) | Softmax (ms) | AV (ms) | Total (ms) |
|---:|---------:|----------:|-------------:|--------:|-----------:|
| 64 | 15.9581 | 2.7437 | 0.0759 | 2.6820 | 21.4739 |
| 128 | 31.9364 | 10.6394 | 0.2961 | 10.7708 | 53.6964 |
| 256 | 64.9426 | 41.8353 | 1.2293 | 46.0075 | 154.2349 |
| 512 | 129.9347 | 165.3354 | 4.8370 | 182.8377 | 483.7381 |
| 1024 | 313.1942 | 816.6285 | 22.5716 | 839.0026 | 1994.8646 |

At `N = 1024`, the dominant stages are:

- `QK^T`: ~817 ms
- `AV`: ~839 ms
- `QKV`: ~313 ms
- `Softmax`: ~23 ms

This confirms that the two `N x N` attention matrix multiplications dominate as sequence length increases.

### OpenMP results

The OpenMP implementation was validated against the CPU reference with the following settings:

- Sequence length: 64
- Input dimension: 32
- Attention dimension: 32
- Value dimension: 32
- OpenMP threads: 2
- Maximum absolute error: 0

| Threads | N | Total (ms) |
|--------:|---:|-----------:|
| 1 | 64 | 20.4965 |
| 1 | 128 | 54.0243 |
| 1 | 256 | 144.6017 |
| 1 | 512 | 464.1605 |
| 1 | 1024 | 1792.1684 |
| 2 | 64 | 15.5722 |
| 2 | 128 | 38.3959 |
| 2 | 256 | 113.2897 |
| 2 | 512 | 600.0738 |
| 2 | 1024 | 1270.2798 |

Approximate 2-thread speedups:

| N | Speedup |
|---:|--------:|
| 64 | 1.32x |
| 128 | 1.41x |
| 256 | 1.28x |
| 512 | 0.77x |
| 1024 | 1.41x |

The `N = 512` case became slower with two threads, indicating that parallel overhead and CPU topology can dominate the benefit of parallelism.

### CUDA baseline

| N | GPU time (ms) | GFLOPS | Max error |
|---:|--------------:|-------:|---------:|
| 64 | 0.0659 | 127.3780 | 4.6492e-05 |
| 128 | 0.1109 | 189.0280 | 7.0095e-05 |
| 256 | 0.2534 | 231.7222 | 7.4744e-05 |
| 512 | 0.7054 | 261.6326 | 8.7738e-05 |
| 1024 | 2.1711 | 293.6495 | 9.3460e-05 |

These timings represent GPU computation and should not be confused with end-to-end host/device transfer costs.

### CUDA end-to-end measurement

A separate benchmark measures host-to-device transfer, GPU computation, and device-to-host transfer.

For `N = 1024`:

- GPU compute: `2.2377 ms`
- End-to-end: `2.6612 ms`
- Difference: `0.4235 ms`
- Share of end-to-end time: approximately `15.9%`

This illustrates why kernel time and application-level latency should be reported separately. The compute and end-to-end measurements are independent benchmarks.

### CUDA stage profiling

A CUDA-event-based profiler was used to estimate the contribution of each stage.

For `N = 1024`, the profiled breakdown was:

| Stage | Time (ms) | Share |
|---|---:|---:|
| QKV | 0.3809 | 12.12% |
| Transpose | 0.0123 | 0.39% |
| QK^T | 0.9400 | 29.90% |
| Scale | 0.0396 | 1.26% |
| Softmax | 0.7797 | 24.80% |
| AV | 0.9729 | 30.95% |
| Total | 3.1436 | 100% |

The profiling build adds CUDA event/timing overhead, so the profiled total should not be directly compared with the unprofiled CUDA baseline GPU computation time. The profiler is primarily used to identify relative stage contributions and bottlenecks.

The major optimization targets identified were therefore:

- `AV`
- `QK^T`
- `Softmax`

### CUDA optimization 1: Shared-memory tiled GEMM

The tiled implementation uses:

- `16 x 16` thread blocks
- Shared-memory `A` tile
- Shared-memory `B` tile
- One output element per thread

| N | Baseline (ms) | Tiled (ms) | Speedup |
|---:|--------------:|-----------:|--------:|
| 64 | 0.0659 | 0.0743 | 0.89x |
| 128 | 0.1109 | 0.1165 | 0.95x |
| 256 | 0.2534 | 0.2553 | 0.99x |
| 512 | 0.7054 | 0.6840 | 1.03x |
| 1024 | 2.1711 | 2.0314 | 1.07x |

For `N = 1024`, the tiled version reduced latency by approximately `6.4%`.

The optimization was more useful for larger matrices. For small matrices, shared-memory setup and synchronization overhead offset the benefit of data reuse.

This experiment is retained as an important negative/limited result rather than being omitted.

### CUDA optimization 2: Parallel row-wise softmax

The original implementation assigned one CUDA thread to each row and performed all work sequentially. The optimized version uses cooperative threads with reductions across the row.

| N | Baseline (ms) | Parallel Softmax (ms) | Speedup |
|---:|--------------:|----------------------:|--------:|
| 64 | 0.0659 | 0.0581 | 1.13x |
| 128 | 0.1109 | 0.1004 | 1.10x |
| 256 | 0.2534 | 0.1986 | 1.28x |
| 512 | 0.7054 | 0.5119 | 1.38x |
| 1024 | 2.1711 | 1.6056 | 1.35x |

At `N = 1024`, the optimized softmax version reduced latency by approximately `26%` while preserving correctness.

This is the strongest targeted optimization obtained so far.

## Key Findings

1. Sequence length strongly affects attention cost.
2. CPU parallelism is not automatically beneficial.
3. GPU acceleration is substantial.
4. Simple tiling has limited benefit.
5. Parallelizing softmax produced a significant improvement.

## Reproducibility

### Google Colab

1. Open a new Google Colab notebook.
2. Set runtime to a T4 GPU.
3. Verify the GPU:

```bash
!nvidia-smi
```

4. Verify CUDA:

```bash
!nvcc --version
```

5. Clone the repository:

```bash
!git clone https://github.com/LokeshR926/attension-project.git
%cd attension-project
```

6. Build the desired implementation using the commands in this README.

Example:

```bash
!nvcc -O2 -std=c++17 cuda/self_attention.cu \
    -o cuda/self_attention

!./cuda/self_attention
```

### Kaggle

The same source can be reproduced using a Kaggle notebook with GPU acceleration.

1. Create a Kaggle notebook.
2. Enable GPU acceleration.
3. Clone the repository:

```bash
!git clone https://github.com/LokeshR926/attension-project.git
%cd attension-project
```

4. Verify the GPU with `nvidia-smi`.
5. Compile using `nvcc`.
6. Run the tests and benchmarks.

### Linux + NVIDIA GPU

Requirements:

- Linux
- NVIDIA GPU
- NVIDIA driver
- CUDA Toolkit
- C++17 compiler

Verify:

```bash
nvidia-smi
nvcc --version
g++ --version
```

Then compile and execute the desired experiments using the commands above.

## Reproduction Checklist

A reproduction should verify the following:

- [ ] GPU/CPU environment recorded
- [ ] Source repository cloned
- [ ] CPU tests pass
- [ ] OpenMP correctness test passes
- [ ] CUDA baseline compiles
- [ ] CUDA baseline correctness passes
- [ ] CUDA tiled version compiles
- [ ] CUDA softmax version compiles
- [ ] `N = 64, 128, 256, 512, 1024` tested
- [ ] Five repetitions used
- [ ] Results recorded
- [ ] Maximum error checked
- [ ] Results compared with reference measurements

A reproduction is considered successful even if absolute execution times differ, provided that:

- the implementation produces correct output within the expected tolerance,
- the experiments run successfully,
- the performance trends are broadly consistent, and
- the hardware/software environment is documented.

## Reproduction Expectations

The results included in this repository are reference measurements from the stated hardware and software environment.

- **On the same Tesla T4 / similar software environment**, results should be reasonably close to the reference measurements.
- **On different GPUs or CPUs**, exact timings are not expected to match.
- **Primary reproduction criteria** are correctness (output within expected tolerance), successful execution, and broad consistency in performance trends rather than bit-for-bit identical timing.

For example, the reference T4 result for the parallel-softmax version at `N = 1024` is `1.6056 ms`. Another T4 session may produce a somewhat different value because of:

- GPU clock frequency
- thermal state
- Colab resource allocation
- background activity
- CUDA/runtime version
- benchmark repetition variation

Therefore, reproduction should focus on both correctness and performance trends.

## Current Project Status

| Component | Status |
|---|---|
| Matrix abstraction | Complete |
| Matrix multiplication | Complete |
| Matrix transpose | Complete |
| Stable softmax | Complete |
| CPU attention | Complete |
| CPU correctness tests | Complete |
| CPU benchmark | Complete |
| OpenMP implementation | Complete |
| OpenMP correctness | Complete |
| OpenMP benchmark | Complete |
| CUDA hello world | Complete |
| CUDA vector addition | Complete |
| CUDA baseline attention | Complete |
| CUDA correctness | Complete |
| T4 baseline benchmark | Complete |
| End-to-end GPU benchmark | Complete |
| CUDA stage profiling | Complete |
| Shared-memory tiled GEMM | Complete |
| Parallel softmax optimization | Complete |
| Final documentation | Complete |
| H200 comparison | Not yet performed |
| Further GEMM optimization | Optional future work |

## Limitations

- The benchmark uses a simplified attention implementation.
- Performance measurements are environment-specific.
- The OpenMP CPU setup is limited by the available Colab hardware topology.
- Shared-memory tiling shows only modest gains for the current kernel design.
- This project is intended to study performance trends rather than serve as a production-grade optimized Transformer implementation.

## License

No explicit open-source license is currently specified for this repository. If the project is later released under a specific license, this section should be updated accordingly.

## Summary

This project provides a simple but complete study of self-attention performance on CPU and GPU systems. It offers a baseline CPU implementation, an OpenMP alternative, a CUDA implementation, and several optimization experiments that highlight where the dominant bottlenecks occur and how they can be addressed in a reproducible way.
