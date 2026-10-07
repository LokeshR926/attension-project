Transformer Self-Attention Performance Study

A reproducible implementation and performance study of Transformer self-attention using sequential CPU, OpenMP, and CUDA implementations.

The project focuses on understanding where self-attention spends computation, how parallel execution changes performance, and how targeted CUDA optimizations affect end-to-end latency.

The current GPU experiments were performed on an NVIDIA Tesla T4 in Google Colab.

Important: This is a performance-engineering/implementation study. The reported timings are hardware- and environment-dependent. The supplied results are reference measurements from the stated environment; reproducing the experiments on another GPU should reproduce the methodology and correctness, but not necessarily the exact timings.

1. Objectives

The project investigates self-attention from a systems and performance perspective.

Main objectives:

Implement a sequential CPU reference.

Implement an OpenMP parallel CPU version.

Implement a CUDA GPU version.

Validate parallel implementations against the CPU reference.

Measure execution time as sequence length increases.

Profile the individual attention stages.

Identify performance bottlenecks.

Test targeted CUDA optimizations.

Record both successful and unsuccessful optimization attempts.

Make the experiments reproducible on Linux, Google Colab, and other CUDA-capable environments.

The current study intentionally uses a simple implementation so that the effect of individual optimizations can be isolated and understood.

2. Mathematical Formulation

For an input matrix

X ∈ R^(N × D)

the query, key, and value matrices are

Q = XW_Q
K = XW_K
V = XW_V

Scaled dot-product self-attention is

Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V

The implementation consists of these main stages:

QKV projection

Transpose of K

QK^T

Scaling by 1 / sqrt(d_k)

Row-wise softmax

Attention-weighted V (AV)

The major matrix-multiplication stages have approximately O(N^2 d) computational complexity, while row-wise softmax has approximately O(N^2) work.

3. Project Structure

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

Compiled executables and object files are intentionally not required in the repository. They should be generated locally using the supplied source code.

4. Software and Hardware

CPU development environment

The CPU experiments were performed in Google Colab on an Intel Xeon environment exposed as:

CPU(s):             2
Model name:        Intel(R) Xeon(R) CPU @ 2.00GHz
Thread(s)/core:    2
Core(s)/socket:    1

Therefore, the OpenMP experiment should not be interpreted as scaling across a large multicore CPU. The Colab runtime exposed one physical core with two logical threads.

GPU reference environment

GPU:       NVIDIA Tesla T4
VRAM:      15360 MiB
Driver:    580.82.07
CUDA:      13.0
Precision: FP32

Compiler

C++17 and NVIDIA nvcc are used.

5. Building the CPU Implementation

From the project root:

g++ -std=c++17 -O2 \
    -Iinclude \
    src/matrix.cpp \
    src/matrix_ops.cpp \
    src/attention.cpp \
    benchmarks/cpu_attention.cpp \
    -o benchmarks/cpu_attention

Run:

./benchmarks/cpu_attention

6. Building the OpenMP Implementation

Compile:

g++ -std=c++17 -O2 -fopenmp \
    -Iinclude \
    src/matrix.cpp \
    src/matrix_ops.cpp \
    src/attention_openmp.cpp \
    benchmarks/openmp_attention.cpp \
    -o benchmarks/openmp_attention

Run:

./benchmarks/openmp_attention

The OpenMP implementation parallelizes matrix multiplication and row-wise operations while retaining the sequential implementation as the correctness reference.

7. Building the CUDA Implementations

CUDA baseline:

nvcc -O2 -std=c++17 \
    cuda/self_attention.cu \
    -o cuda/self_attention

End-to-end CUDA benchmark:

nvcc -O2 -std=c++17 \
    cuda/self_attention_e2e.cu \
    -o cuda/self_attention_e2e

Stage profiler:

nvcc -O2 -std=c++17 \
    cuda/self_attention_profile.cu \
    -o cuda/self_attention_profile

Tiled GEMM experiment:

nvcc -O2 -std=c++17 \
    cuda/self_attention_tiled.cu \
    -o cuda/self_attention_tiled

Parallel-softmax experiment:

nvcc -O2 -std=c++17 \
    cuda/self_attention_softmax.cu \
    -o cuda/self_attention_softmax

8. Correctness Validation

The project contains tests for:

Matrix construction and indexing

Matrix multiplication

Matrix transpose

Numerically stable softmax

Scaled dot-product attention

Complete self-attention

OpenMP attention

The GPU implementations compare their output against the CPU reference.

For the T4 experiments, the maximum absolute error remained below:

1e-4

Representative final values:

N = 64    → 4.6492e-05
N = 128   → 7.0095e-05
N = 256   → 7.4744e-05
N = 512   → 8.7738e-05
N = 1024  → 9.3460e-05

Small differences are expected because CPU and GPU floating-point operations can accumulate values in different orders.

9. Numerical Stability

Softmax is implemented using the numerically stable form:

softmax(x_i) =
    exp(x_i - max(x)) /
    sum_j exp(x_j - max(x))

Subtracting the maximum value prevents large positive values from causing exponential overflow.

The implementation was tested with large input values around 1000.

10. Experimental Configuration

Unless otherwise stated:

D   = 128
d_k = 128
d_v = 128
N   = 64, 128, 256, 512, 1024
Repetitions = 5
Datatype = float32

The benchmark uses deterministic input generation so that different implementations can be compared consistently.

The reference CPU and OpenMP measurements are stored under `results/`. The T4 baseline and tiled-GEMM measurements are stored under `results/cuda/`. The latest end-to-end and stage-profiler measurements reported below were recorded from the corresponding T4 Colab benchmark runs.

11. Sequential CPU Results

Reference measurements:

N

QKV (ms)

QK^T (ms)

Softmax (ms)

AV (ms)

Total (ms)

64

15.9581

2.7437

0.0759

2.6820

21.4739

128

31.9364

10.6394

0.2961

10.7708

53.6964

256

64.9426

41.8353

1.2293

46.0075

154.2349

512

129.9347

165.3354

4.8370

182.8377

483.7381

1024

313.1942

816.6285

22.5716

839.0026

1994.8646

At N = 1024, the dominant stages are:

QK^T: approximately 817 ms

AV: approximately 839 ms

QKV: approximately 313 ms

Softmax: approximately 23 ms

The CPU implementation therefore shows that the two N × N attention matrix multiplications dominate as sequence length grows.

Note: the CPU QK^T timing includes the K transpose operation.

12. OpenMP Results

The OpenMP implementation was validated against the CPU reference with:

Sequence length: 64
Input dimension: 32
Attention dimension: 32
Value dimension: 32
OpenMP threads: 2
Maximum absolute error: 0

The 1-thread and 2-thread benchmark results were:

1 thread

N

Total (ms)

64

20.4965

128

54.0243

256

144.6017

512

464.1605

1024

1792.1684

2 threads

N

Total (ms)

64

15.5722

128

38.3959

256

113.2897

512

600.0738

1024

1270.2798

The corresponding 2-thread speedups over the 1-thread OpenMP run were approximately:

N

Speedup

64

1.32×

128

1.41×

256

1.28×

512

0.77×

1024

1.41×

The N=512 case became slower with two threads, demonstrating that parallelization overhead and the characteristics of the available Colab CPU can affect scaling.

13. CUDA Baseline

The baseline CUDA implementation uses a simple one-output-element-per-thread matrix multiplication kernel.

The benchmark was run on the Tesla T4.

N

GPU time (ms)

GFLOPS

Max error

64

0.0659

127.3780

4.6492e-05

128

0.1109

189.0280

7.0095e-05

256

0.2534

231.7222

7.4744e-05

512

0.7054

261.6326

8.7738e-05

1024

2.1711

293.6495

9.3460e-05

These timings represent the GPU computation benchmark and should not be confused with end-to-end host/device transfer timing.

14. CUDA End-to-End Measurement

A separate benchmark measures:

Host → Device transfer
        +
GPU computation
        +
Device → Host transfer

For N=1024:

GPU compute:       2.2377 ms
End-to-end:        2.6612 ms

The difference is approximately:

0.4235 ms

or about:

15.9%

of the measured end-to-end time.

This demonstrates why GPU kernel time and application-level end-to-end latency should be reported separately.

15. CUDA Stage-Level Profiling

A CUDA-event-based profiler was used to estimate the contribution of each stage.

For N=1024, the profiled breakdown was:

Stage

Time

Share

QKV

0.3809 ms

12.12%

Transpose

0.0123 ms

0.39%

QK^T

0.9400 ms

29.90%

Scale

0.0396 ms

1.26%

Softmax

0.7797 ms

24.80%

AV

0.9729 ms

30.95%

Total

3.1436 ms

100%

The profiling build uses a different measurement path and adds CUDA-event timing overhead. Therefore, the profiled total should not be treated as a directly comparable replacement for the unprofiled CUDA compute benchmark. The profiler is used primarily to identify relative stage contributions and bottlenecks.

The major optimization targets identified were therefore:

AV

QK^T

Softmax

Transpose and scaling were too small to be attractive independent optimization targets.

16. CUDA Optimization 1: Shared-Memory Tiled GEMM

A separate copy of the CUDA baseline was created so that the original baseline remained unchanged.

The tiled implementation uses:

16 × 16 thread blocks
Shared-memory A tile
Shared-memory B tile
One output element per thread

T4 results:

N

Baseline (ms)

Tiled (ms)

Speedup

64

0.0659

0.0743

0.89×

128

0.1109

0.1165

0.95×

256

0.2534

0.2553

0.99×

512

0.7054

0.6840

1.03×

1024

2.1711

2.0314

1.07×

At N=1024:

Speedup ≈ 1.07×
Latency reduction ≈ 6.4%

The optimization was more useful for larger matrices. For small matrices, shared-memory setup and synchronization overhead offset the benefit of data reuse.

This experiment is retained as an important negative/limited result rather than being omitted.

17. CUDA Optimization 2: Parallel Row-Wise Softmax

The original softmax assigned one CUDA thread to each row and performed all work for that row sequentially.

The optimized version uses:

256 threads per row/block
        ↓
parallel maximum reduction
        ↓
parallel exponential/sum calculation
        ↓
parallel sum reduction
        ↓
parallel normalization

T4 results:

N

Baseline (ms)

Parallel Softmax (ms)

Speedup

64

0.0659

0.0581

1.13×

128

0.1109

0.1004

1.10×

256

0.2534

0.1986

1.28×

512

0.7054

0.5119

1.38×

1024

2.1711

1.6056

1.35×

At N=1024:

Baseline:          2.1711 ms
Optimized:         1.6056 ms
Speedup:           1.35×
Latency reduction: approximately 26%

Correctness remained within the same numerical error range.

This is the strongest targeted optimization obtained so far in the recorded T4 experiments. The values above were recorded from the T4 Colab run; the corresponding source is `cuda/self_attention_softmax.cu`.

18. What the Experiments Show

Several conclusions can be drawn from the current experiments.

18.1 Sequence length strongly affects attention cost

The QK^T and AV operations grow with the square of sequence length, making them increasingly dominant at larger N.

18.2 CPU parallelism is not automatically beneficial

OpenMP provided useful speedups for several sizes, but the N=512 case became slower with two threads. Parallel overhead, CPU topology, scheduling, and resource contention all matter.

18.3 GPU acceleration is substantial

The T4 completes the tested attention workload in milliseconds rather than the hundreds or thousands of milliseconds seen in the simple CPU implementation.

18.4 Simple tiling has limited benefit

Shared-memory tiling produced only a modest improvement for the current matrix sizes and kernel design.

This demonstrates that simply adding shared memory does not automatically produce an optimized GEMM.

18.5 Parallelizing softmax produced a significant improvement

The original row-wise softmax underutilized the GPU because one thread processed an entire row. Cooperating threads and reductions substantially improved the overall benchmark.

19. Reproducibility

The repository is intended to allow another user to reproduce the experiments without relying on the original development environment.

19.1 Google Colab

Open a new Google Colab notebook.

Select:

Runtime → Change runtime type → T4 GPU

Verify the GPU:

!nvidia-smi

Verify CUDA:

!nvcc --version

Clone the repository:

!git clone https://github.com/LokeshR926/attension-project.git
%cd attension-project

Build the desired implementation using the commands in this README.

For example:

!nvcc -O2 -std=c++17 cuda/self_attention.cu \
    -o cuda/self_attention

Run:

!./cuda/self_attention

For the optimized softmax:

!nvcc -O2 -std=c++17 cuda/self_attention_softmax.cu \
    -o cuda/self_attention_softmax

!./cuda/self_attention_softmax

The exact runtime may differ between Colab sessions.

19.2 Kaggle

The same source can be reproduced using a Kaggle Notebook with GPU acceleration.

Create a Kaggle Notebook.

Enable GPU acceleration.

Clone the GitHub repository:

!git clone https://github.com/LokeshR926/attension-project.git
%cd attension-project

Verify the GPU:

!nvidia-smi

Compile using nvcc.

Run the tests and benchmarks.

Kaggle may provide a different GPU model or software stack, so exact timings will differ from the T4 reference results.

19.3 Linux + NVIDIA GPU

Requirements:

Linux

NVIDIA GPU

NVIDIA driver

CUDA Toolkit

C++17 compiler

Verify:

nvidia-smi
nvcc --version
g++ --version

Then compile and execute the desired experiments using the commands above.

20. Reproduction Checklist

A reproduction should verify the following:

[ ] GPU/CPU environment recorded
[ ] Source repository cloned
[ ] CPU tests pass
[ ] OpenMP correctness test passes
[ ] CUDA baseline compiles
[ ] CUDA baseline correctness passes
[ ] CUDA tiled version compiles
[ ] CUDA softmax version compiles
[ ] N = 64, 128, 256, 512, 1024 tested
[ ] Five repetitions used
[ ] Results recorded
[ ] Maximum error checked
[ ] Results compared with reference measurements

A reproduction is considered successful even if absolute execution times differ, provided that:

the implementation produces correct output within the expected tolerance,

the experiments run successfully,

the performance trends are broadly consistent, and

the hardware/software environment is documented.

21. Reference Results vs Reproduced Results

The results included in this repository are reference measurements, not guaranteed performance targets.

For example, the reference T4 result for the parallel-softmax version at N=1024 is:

1.6056 ms

Another T4 session may produce a somewhat different value because of:

GPU clock frequency

thermal state

Colab resource allocation

background activity

CUDA/runtime version

benchmark repetition variation

Therefore, reproduction should focus on both correctness and performance trends, rather than requiring bit-for-bit identical timing.

22. Current Project Status

Component

Status

Matrix abstraction

Complete

Matrix multiplication

Complete

Matrix transpose

Complete

Stable softmax

Complete

CPU attention

Complete

CPU correctness tests

Complete

CPU benchmark

Complete

OpenMP implementation

Complete

OpenMP correctness

Complete

OpenMP benchmark

Complete

CUDA hello world

Complete

CUDA vector addition

Complete

CUDA baseline attention

Complete

CUDA correctness

Complete

T4 baseline benchmark

Complete

End-to-end GPU benchmark

Complete

CUDA stage profiling

Complete

Shared-memory tiled GEMM

Complete

Parallel softmax optimization

Complete

Final documentation

Complete

H200 comparison

Not yet performed

Further GEMM optimization

Optional future work

23. Limitations

The current implementation is intentionally educational and experimental.

It does not attempt to reproduce the performance of production Transformer kernels such as highly optimized GEMM libraries or FlashAttention implementations.

Current limitations include:

FP32 only

Single-head/simple matrix formulation

No batching

No causal masking

No multi-head attention implementation

No Tensor Core implementation

No fused attention kernel

No FlashAttention-style IO-aware algorithm

Simple CUDA GEMM

Limited GPU configurations

T4 reference results only so far

OpenMP evaluation performed on a constrained Colab CPU

These limitations define potential future experiments rather than invalidating the current study.

24. Future Work

Possible extensions include:

More advanced GEMM tiling.

Register blocking.

Loop unrolling and vectorized memory access.

Tensor Core/WMMA experiments.

Fused attention kernels.

FlashAttention-style memory-aware algorithms.

FP16/BF16 experiments.

Larger sequence lengths.

Larger matrix dimensions.

H200 comparison.

Nsight Compute analysis.

Memory bandwidth and occupancy analysis.

These are future directions and are not represented as completed experiments in the current results.

25. Conclusion

This project demonstrates a complete progression from a simple sequential self-attention implementation to CPU parallelism and CUDA GPU execution.

The experiments show that:

self-attention becomes increasingly dominated by N × N operations as sequence length grows;

OpenMP can improve CPU performance, but scaling depends strongly on the available hardware;

GPU execution provides a large reduction in execution time for the tested workload;

simple shared-memory GEMM provides only a modest improvement;

targeted parallelization of row-wise softmax provides a substantial improvement, reaching approximately 1.35× end-to-end speedup at N=1024 on the reference T4 run.

The main lesson of the project is that performance optimization should be driven by measurement and bottleneck analysis, rather than by applying an optimization without understanding where execution time is being spent.
