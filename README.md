#  Mini AI Inference Runtime (C++, CPU)

A lightweight AI inference runtime built in C++ that simulates how modern frameworks (e.g., ONNX Runtime / PyTorch) execute computational graphs under memory and latency constraints.

---

##  Key Features

-  **DAG-based execution engine**
-  **Modular operator abstraction (MatMul, ReLU)**
-  **Custom memory management with buffer reuse**
-  **Per-operator latency profiling**
-  **Benchmarking framework for latency & memory trade-offs**

---

##  System Design

This project implements a simplified inference runtime with:

- **Tensor abstraction** for data representation
- **Operator interface** for extensible computation
- **Graph (DAG)** for modeling computation dependencies
- **Topological execution engine** for correct scheduling
- **Memory pool** for buffer reuse across non-overlapping nodes

---

##  Execution & Profiling

![DAG Execution](images/dag_execution_and_operator_profiling.png)

- Executes operators in **topological order**
- Tracks **per-operator latency**
- Reports:
  - total latency
  - execution order
  - memory usage
  - allocation statistics

---

##  Benchmark Results

### Hidden Dimension = 128

![Benchmark 128](images/memory_optimization_comparison_h128.png)

### Hidden Dimension = 256

![Benchmark 256](images/memory_optimization_comparison_h256.png)

---

##  Key Insights

### Memory Optimization

- Naive execution:
  - Memory scales **linearly** with graph depth
- Buffer reuse:
  - Memory remains **nearly constant**

Example (depth = 32):

| Mode          | Peak Memory | Allocations |
|--------------|------------|------------|
| Naive        | 64 KB      | 64         |
| Buffer Reuse | ~2 KB      | 2          |

---

### Allocation Reduction

- Reduced allocations from **O(N)** → **O(1)**
- Achieved via **buffer reuse across non-overlapping nodes**

---

### Latency Trade-off

- Slight overhead introduced by memory management
- Demonstrates real-world trade-off:
  
> Lower memory footprint vs. additional runtime coordination cost

---

## Build & Run

### Build

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
### Run Demo
./Release/runtime_demo --graph=4 --hidden=64
### Run Benchmark
./Release/runtime_benchmark --hidden=128 --runs=5
### Run Tests
./Release/runtime_tests
## Project Structure
mini_ai_inference_runtime/
├── include/
├── src/
│   ├── graph.cpp
│   ├── execution_engine.cpp
│   └── operators/
├── apps/
│   ├── runtime_demo.cpp
│   ├── runtime_benchmark.cpp
├── tests/
├── images/
└── CMakeLists.txt
## What This Project Demonstrates

This project showcases:

Systems-level understanding of ML inference pipelines
Ability to design efficient execution engines
Practical handling of memory optimization
Performance evaluation via benchmarking and profiling
## Future Improvements
Multi-threaded execution
Operator fusion
GPU backend (CUDA)
Dynamic shape support
Graph optimization passes
## Interview Pitch (TL;DR)

I built a minimal AI inference runtime in C++ that executes DAG-based computation graphs with custom memory optimization and profiling.
Benchmarks showed that buffer reuse reduces memory usage from linear growth to near-constant, while maintaining competitive latency.