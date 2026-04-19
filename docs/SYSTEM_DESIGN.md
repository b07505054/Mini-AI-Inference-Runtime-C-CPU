# System Design Notes

## Core components

### Tensor
A tensor stores:
- shape metadata
- a shared pointer to a buffer
- a flag indicating whether the buffer came from the runtime memory pool

### Operator
Each operator exposes:
- `type()`
- `inferOutputShape(inputs)`
- `execute(inputs, output)`

This keeps the execution engine generic and operator-independent.

### Graph
A graph contains:
- initial values (inputs, weights, constants)
- operator nodes
- output tensor names

Nodes are stored independently from execution order. The scheduler computes the real order later through topological sorting.

### Execution engine
The engine:
1. builds a topological order
2. computes last-use information
3. materializes inputs for each node
4. allocates output buffers from the pool
5. executes operators
6. releases dead activations when reuse is enabled
7. records latency and memory statistics

### Memory pool
The pool maintains a list of reusable buffers. When a new tensor is needed, it tries to find the smallest available compatible buffer. If none exists, it allocates a new one.

## Why this design is interview-friendly
It demonstrates:
- dataflow execution
- dependency resolution
- lifecycle-aware memory management
- instrumentation and benchmarking
- clean C++ interfaces
