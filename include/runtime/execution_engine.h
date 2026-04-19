#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/graph.h"
#include "runtime/memory_pool.h"

namespace runtime {

struct OperatorProfile {
    std::string node_name;
    std::string op_type;
    double latency_ms = 0.0;
};

struct ExecutionOptions {
    bool enable_buffer_reuse = true;
};

struct ExecutionResult {
    std::vector<std::string> execution_order;
    std::vector<OperatorProfile> operator_profiles;
    double total_latency_ms = 0.0;
    MemoryStats memory_stats;
    std::unordered_map<std::string, Tensor> outputs;
};

class ExecutionEngine {
public:
    explicit ExecutionEngine(ExecutionOptions options = {}) : options_(options) {}

    ExecutionResult run(const Graph& graph) const;

private:
    std::unordered_map<std::string, int> computeLastUse(
        const Graph& graph,
        const std::vector<int>& topo_order) const;

    ExecutionOptions options_;
};

}  // namespace runtime
