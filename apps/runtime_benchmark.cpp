#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "runtime/execution_engine.h"
#include "runtime/graph.h"
#include "runtime/operators/matmul.h"
#include "runtime/operators/relu.h"

namespace {

runtime::Tensor makeRandomTensor(const std::vector<int>& shape, int seed_offset = 0) {
    const auto elements = runtime::Tensor::computeNumel(shape);
    std::vector<float> values(elements);

    std::mt19937 rng(17 + seed_offset);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& value : values) {
        value = dist(rng);
    }

    return runtime::Tensor::fromVector(shape, values);
}

runtime::Graph buildBenchmarkGraph(int depth, int hidden_dim) {
    runtime::Graph graph;
    graph.addValue("input", makeRandomTensor({1, hidden_dim}, 0));

    auto matmul = std::make_shared<runtime::MatMul>();
    auto relu = std::make_shared<runtime::ReLU>();
    std::vector<runtime::Node> pending_nodes;

    std::string previous = "input";
    for (int layer = 0; layer < depth; ++layer) {
        const std::string weight_name = "weight_" + std::to_string(layer);
        graph.addValue(weight_name, makeRandomTensor({hidden_dim, hidden_dim}, layer + 1));

        const std::string matmul_output = "matmul_out_" + std::to_string(layer);
        const std::string relu_output = "relu_out_" + std::to_string(layer);

        pending_nodes.push_back(runtime::Node{
            "matmul_" + std::to_string(layer),
            matmul,
            {previous, weight_name},
            matmul_output
        });
        pending_nodes.push_back(runtime::Node{
            "relu_" + std::to_string(layer),
            relu,
            {matmul_output},
            relu_output
        });

        previous = relu_output;
    }

    for (auto it = pending_nodes.rbegin(); it != pending_nodes.rend(); ++it) {
        graph.addNode(*it);
    }

    graph.setOutputs({previous});
    return graph;
}

struct AggregateMetrics {
    double avg_latency_ms = 0.0;
    double avg_peak_active_kb = 0.0;
    double avg_peak_reserved_kb = 0.0;
    double avg_new_allocations = 0.0;
    double avg_buffer_reuses = 0.0;
};

AggregateMetrics benchmark(int depth, int hidden_dim, bool reuse, int runs) {
    runtime::ExecutionEngine engine(runtime::ExecutionOptions{reuse});
    AggregateMetrics metrics;

    for (int run = 0; run < runs; ++run) {
        auto graph = buildBenchmarkGraph(depth, hidden_dim);
        auto result = engine.run(graph);
        metrics.avg_latency_ms += result.total_latency_ms;
        metrics.avg_peak_active_kb += static_cast<double>(result.memory_stats.peak_active_bytes) / 1024.0;
        metrics.avg_peak_reserved_kb += static_cast<double>(result.memory_stats.peak_reserved_bytes) / 1024.0;
        metrics.avg_new_allocations += static_cast<double>(result.memory_stats.new_allocations);
        metrics.avg_buffer_reuses += static_cast<double>(result.memory_stats.buffer_reuses);
    }

    metrics.avg_latency_ms /= static_cast<double>(runs);
    metrics.avg_peak_active_kb /= static_cast<double>(runs);
    metrics.avg_peak_reserved_kb /= static_cast<double>(runs);
    metrics.avg_new_allocations /= static_cast<double>(runs);
    metrics.avg_buffer_reuses /= static_cast<double>(runs);
    return metrics;
}

}  // namespace

int main(int argc, char** argv) {
    int hidden_dim = 128;
    int runs = 5;
    std::vector<int> depths = {2, 4, 8, 16, 32};

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--hidden=", 0) == 0) {
            hidden_dim = std::stoi(arg.substr(9));
        } else if (arg.rfind("--runs=", 0) == 0) {
            runs = std::stoi(arg.substr(7));
        }
    }

    std::cout << "Mini AI Inference Runtime Benchmark\n";
    std::cout << "===============================================================\n";
    std::cout << "Hidden dimension: " << hidden_dim << ", runs per config: " << runs << "\n\n";
    std::cout << std::left
              << std::setw(10) << "Depth"
              << std::setw(18) << "Mode"
              << std::setw(18) << "Avg Latency (ms)"
              << std::setw(22) << "Avg Peak Active (KB)"
              << std::setw(24) << "Avg Peak Reserved (KB)"
              << std::setw(18) << "Avg Allocs"
              << std::setw(18) << "Avg Reuses"
              << "\n";

    for (int depth : depths) {
        const auto naive = benchmark(depth, hidden_dim, false, runs);
        const auto reuse = benchmark(depth, hidden_dim, true, runs);

        std::cout << std::setw(10) << depth
                  << std::setw(18) << "naive"
                  << std::setw(18) << std::fixed << std::setprecision(3) << naive.avg_latency_ms
                  << std::setw(22) << naive.avg_peak_active_kb
                  << std::setw(24) << naive.avg_peak_reserved_kb
                  << std::setw(18) << naive.avg_new_allocations
                  << std::setw(18) << naive.avg_buffer_reuses
                  << "\n";

        std::cout << std::setw(10) << depth
                  << std::setw(18) << "buffer_reuse"
                  << std::setw(18) << reuse.avg_latency_ms
                  << std::setw(22) << reuse.avg_peak_active_kb
                  << std::setw(24) << reuse.avg_peak_reserved_kb
                  << std::setw(18) << reuse.avg_new_allocations
                  << std::setw(18) << reuse.avg_buffer_reuses
                  << "\n";
    }

    return 0;
}
