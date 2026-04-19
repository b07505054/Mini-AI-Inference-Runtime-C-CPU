#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "runtime/execution_engine.h"
#include "runtime/graph.h"
#include "runtime/operators/matmul.h"
#include "runtime/operators/relu.h"

namespace {

runtime::Tensor makeRandomTensor(const std::vector<int>& shape, float min_value = -1.0f, float max_value = 1.0f) {
    const auto elements = runtime::Tensor::computeNumel(shape);
    std::vector<float> values(elements);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(min_value, max_value);

    for (auto& value : values) {
        value = dist(rng);
    }

    return runtime::Tensor::fromVector(shape, values);
}

runtime::Graph buildDemoGraph(int depth, int hidden_dim) {
    runtime::Graph graph;
    graph.addValue("input", makeRandomTensor({1, hidden_dim}));

    std::string previous = "input";
    auto matmul = std::make_shared<runtime::MatMul>();
    auto relu = std::make_shared<runtime::ReLU>();
    std::vector<runtime::Node> pending_nodes;

    for (int layer = 0; layer < depth; ++layer) {
        const std::string weight_name = "weight_" + std::to_string(layer);
        graph.addValue(weight_name, makeRandomTensor({hidden_dim, hidden_dim}));

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

std::string join(const std::vector<std::string>& items, const std::string& delimiter) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i != 0) {
            oss << delimiter;
        }
        oss << items[i];
    }
    return oss.str();
}

double checksum(const runtime::Tensor& tensor) {
    double sum = 0.0;
    for (std::size_t i = 0; i < tensor.numel(); ++i) {
        sum += tensor.data()[i];
    }
    return sum;
}

}  // namespace

int main(int argc, char** argv) {
    int depth = 4;
    int hidden_dim = 64;
    bool reuse = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--graph=", 0) == 0) {
            depth = std::stoi(arg.substr(8));
        } else if (arg.rfind("--hidden=", 0) == 0) {
            hidden_dim = std::stoi(arg.substr(9));
        } else if (arg == "--no-reuse") {
            reuse = false;
        }
    }

    auto graph = buildDemoGraph(depth, hidden_dim);
    runtime::ExecutionEngine engine(runtime::ExecutionOptions{reuse});
    const auto result = engine.run(graph);

    std::cout << "Mini AI Inference Runtime Demo\n";
    std::cout << "====================================\n";
    std::cout << "Graph depth: " << depth << " operator pairs\n";
    std::cout << "Hidden dimension: " << hidden_dim << "\n";
    std::cout << "Buffer reuse: " << (reuse ? "enabled" : "disabled") << "\n\n";

    std::cout << "Execution order: " << join(result.execution_order, " -> ") << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Total latency: " << result.total_latency_ms << " ms\n";
    std::cout << "Peak active memory: "
              << static_cast<double>(result.memory_stats.peak_active_bytes) / 1024.0 << " KB\n";
    std::cout << "Peak reserved memory: "
              << static_cast<double>(result.memory_stats.peak_reserved_bytes) / 1024.0 << " KB\n";
    std::cout << "New allocations: " << result.memory_stats.new_allocations << "\n";
    std::cout << "Buffer reuses: " << result.memory_stats.buffer_reuses << "\n\n";

    std::cout << "Per-operator latency:\n";
    for (const auto& profile : result.operator_profiles) {
        std::cout << "  - " << std::setw(10) << std::left << profile.node_name
                  << " (" << std::setw(6) << std::left << profile.op_type << ") : "
                  << profile.latency_ms << " ms\n";
    }

    if (!result.outputs.empty()) {
        const auto& output = result.outputs.begin()->second;
        std::cout << "\nOutput checksum: " << checksum(output) << "\n";
    }

    return 0;
}
