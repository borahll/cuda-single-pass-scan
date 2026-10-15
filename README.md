# Single-Pass Parallel Prefix Scan (CUDA)

A CUDA project implementing an efficient **single-pass parallel prefix scan** using the **decoupled look-back** technique proposed by Merrill and Garland.

Developed as part of the **Programming Massively Parallel Hardware (PMPH)** course at the University of Copenhagen, 2026.

## Overview

A prefix scan (also known as a prefix sum when the operation is addition) is a fundamental parallel primitive that computes cumulative results over an array.

For example:

```text
Input:           [3, 1, 7, 0, 4, 1]
Inclusive scan:  [3, 4, 11, 11, 15, 16]
Exclusive scan:  [0, 3, 4, 11, 11, 15]
```

Prefix scans are widely used in parallel computing applications, including:

- Stream compaction
- Radix sort
- Histogram processing
- Parallel memory allocation
- Graph algorithms

Traditional GPU prefix scan implementations typically require multiple passes over the input data to compute block-level aggregates and distribute global prefixes.

The **decoupled look-back** algorithm addresses this limitation by allowing thread blocks to compute their local scans and determine their global offsets within a single device-wide pass, reducing global memory traffic and synchronization overhead.

## Project Objectives

The primary objective of this project is to implement and evaluate a single-pass prefix scan algorithm using CUDA.

The project focuses on:

- Implementing the decoupled look-back algorithm.
- Efficiently computing block-local prefix scans.
- Coordinating prefix information between CUDA thread blocks.
- Minimizing global memory accesses and synchronization overhead.
- Evaluating correctness against a sequential CPU implementation.
- Analyzing execution time, scalability, and effective memory bandwidth.
- Comparing the single-pass approach with traditional parallel scan algorithms.

## Algorithm

The implementation follows the **single-pass parallel prefix scan with decoupled look-back** approach.

The input array is divided into tiles, with each tile processed by a CUDA thread block.

Each block performs the following operations:

1. **Local Scan:** Compute the prefix scan of the elements assigned to the block.
2. **Block Aggregate:** Calculate the total aggregate of the block's elements.
3. **Publish Aggregate:** Make the aggregate available to other thread blocks.
4. **Decoupled Look-back:** Inspect information published by preceding blocks to determine the global prefix offset.
5. **Global Prefix:** Combine the computed offset with the local scan results.
6. **Output:** Write the final prefix scan values to global memory.

This approach allows blocks to make progress independently while resolving dependencies between different portions of the input.

### Advantages of Decoupled Look-back

The algorithm offers several advantages over traditional multi-pass approaches:

- **Single-pass execution:** Each input element needs to be read once and each output element written once in the ideal algorithm.
- **Reduced memory traffic:** Approximately `2n` element transfers compared with `3n` for conventional reduce-then-scan approaches.
- **Improved parallelism:** Thread blocks can perform useful work while global prefix information propagates.
- **Lower synchronization overhead:** Global prefix propagation is integrated into the scan operation.
- **High memory efficiency:** The algorithm is designed to approach the effective memory bandwidth of a simple data copy.

## Implementation

The implementation is developed in **CUDA C++**, using GPU parallelism to accelerate prefix scan computation.

The main implementation considerations include:

### Intra-block Scan

Each CUDA thread block computes a local prefix scan using parallel scan techniques.

Efficient intra-block computation is essential for achieving high throughput.

### Inter-block Communication

Blocks communicate through global memory to exchange aggregate and prefix information.

The decoupled look-back mechanism allows blocks to determine their global offsets without requiring a separate kernel launch for every stage of the scan.

Correct synchronization and memory ordering are essential to ensure that blocks observe valid prefix information.

### Memory Optimization

The implementation aims to minimize memory transactions and maximize effective bandwidth through:

- Coalesced global memory accesses.
- Efficient use of registers and shared memory.
- Reduced redundant global memory operations.
- Appropriate thread block and tile sizes.

## Performance Evaluation

Performance is evaluated using different input sizes to investigate the scalability and efficiency of the implementation.

The main evaluation metrics are:

**Execution Time**

GPU execution time is measured to evaluate the efficiency of the scan operation.

**Effective Memory Bandwidth**

For an out-of-place scan, effective memory bandwidth is calculated as:

\[
BW = \frac{2 \times N \times \text{sizeof}(T)}{t}
\]

Where:

- \(N\) is the number of input elements.
- \(T\) is the element data type.
- \(t\) is the execution time in seconds.

**Speedup**

Performance improvement over a reference implementation is calculated as:

\[
S = \frac{T_{\text{reference}}}{T_{\text{GPU}}}
\]

**Correctness**

The output is validated against a sequential CPU reference implementation to verify that the parallel scan produces the expected results.

For floating-point data, small numerical differences may occur because parallel computation changes the order of arithmetic operations.

## Requirements

The project requires:

- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit
- CUDA-compatible C++ compiler

## Compilation and Execution

The exact compilation commands depend on the source files and project structure.

A typical CUDA compilation command is:

```bash
nvcc -O3 <source>.cu -o scan
```

The resulting executable can then be run using:

```bash
./scan
```

These commands are illustrative and should be adjusted to match the actual implementation.

## References

1. Merrill, D., & Garland, M. (2016). **Single-pass Parallel Prefix Scan with Decoupled Look-back.** NVIDIA Technical Report NVR-2016-002.  
   https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

2. University of Copenhagen. **Programming Massively Parallel Hardware (PMPH), 2026 — Single-Pass Scan Group Project.**  
   https://github.com/diku-dk/pmph-e2026-pub/blob/main/group-projects/single-pass-scan/single-pass-scan-paper.pdf

3. NVIDIA. **CUB DeviceScan Documentation.**  
   https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceScan.html

## Project Information

**Course:** Programming Massively Parallel Hardware (PMPH)  
**Institution:** University of Copenhagen (DIKU)  
**Year:** 2026  
**Language:** CUDA C++  
**Topic:** GPU Parallel Algorithms — Single-Pass Prefix Scan

---

**Note:** This README describes the algorithm and intended implementation. The exact build instructions, performance measurements, and implementation-specific details should be updated to match the completed project.
