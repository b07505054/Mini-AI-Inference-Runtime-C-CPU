# Interview Q&A

## 1) What problem does this project solve?
This project simulates a small inference runtime for CPU-only environments. Instead of calling a high-level framework, it executes a computation graph directly through a minimal operator abstraction, a DAG scheduler, and a custom memory strategy.

## 2) Why is the graph modeled as a DAG?
Inference graphs are naturally acyclic in the forward pass: each operator consumes tensors produced by earlier nodes. Modeling the graph as a DAG lets the runtime compute a topological order automatically, so execution remains correct even if nodes are inserted out of order.

## 3) How does scheduling work?
The runtime scans each node's input tensors, finds which node produced each one, builds dependency edges, computes indegrees, and runs Kahn's algorithm for topological sorting. The output is a valid execution order for the graph.

## 4) Why implement memory reuse?
Naively allocating a fresh buffer for every intermediate tensor increases allocation overhead and can cause peak memory to scale with graph depth. This runtime tracks the last use of each activation, returns dead tensors to a pool, and reuses compatible buffers later.

## 5) What does “last use” mean?
For each produced tensor, the engine records the final node that consumes it. Once that final consumer finishes execution, the tensor is no longer needed for correctness, so its buffer can be recycled.

## 6) What memory metrics are tracked?
- new allocations
- buffer reuses
- peak active bytes
- peak reserved bytes

Peak active bytes represent the largest amount of activation memory simultaneously live. Peak reserved bytes represent the largest amount of memory held by the runtime allocator.

## 7) Why benchmark naive vs. buffer reuse?
The comparison makes the memory optimization measurable. In the naive mode, intermediate activations accumulate across the graph. In the buffer-reuse mode, the runtime keeps peak activation memory almost constant while graph depth grows.

## 8) What are the current limitations?
- only `MatMul` and `ReLU` are implemented
- CPU only
- no ONNX parser or model loader
- no parallel scheduling for independent branches
- no static compilation or kernel fusion

## 9) How would you extend this project?
- add `Add`, `Softmax`, `LayerNorm`, `Conv2D`
- support graph import from JSON or ONNX
- implement branch-aware parallel scheduling
- add static memory planning before execution
- add quantized kernels for edge-device workloads

## 10) What production systems does this resemble?
Conceptually, it overlaps with the responsibilities of execution engines in systems like ONNX Runtime, TensorRT, or framework runtimes, but this repo is intentionally much smaller and focused on the core ideas needed for interviews and learning.
