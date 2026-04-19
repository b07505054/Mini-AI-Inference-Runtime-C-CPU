#include "runtime/execution_engine.h"

#include <stdexcept>

namespace runtime {

std::unordered_map<std::string, int> ExecutionEngine::computeLastUse(
    const Graph& graph,
    const std::vector<int>& topo_order) const {
    std::unordered_map<std::string, int> last_use;
    std::unordered_map<int, int> position_of_node;
    for (int position = 0; position < static_cast<int>(topo_order.size()); ++position) {
        position_of_node[topo_order[position]] = position;
    }

    for (int position = 0; position < static_cast<int>(topo_order.size()); ++position) {
        const auto& node = graph.nodes()[topo_order[position]];
        for (const auto& input_name : node.inputs) {
            last_use[input_name] = position;
        }
    }

    return last_use;
}

ExecutionResult ExecutionEngine::run(const Graph& graph) const {
    const auto topo_order = graph.topologicalOrder();
    const auto last_use = computeLastUse(graph, topo_order);

    MemoryPool pool(options_.enable_buffer_reuse);

    std::unordered_map<std::string, Tensor> live_tensors = graph.initialValues();
    ExecutionResult result;
    result.execution_order.reserve(topo_order.size());

    const auto total_start = std::chrono::high_resolution_clock::now();

    for (int position = 0; position < static_cast<int>(topo_order.size()); ++position) {
        const auto& node = graph.nodes()[topo_order[position]];
        result.execution_order.push_back(node.name);

        std::vector<Tensor> inputs;
        inputs.reserve(node.inputs.size());
        for (const auto& input_name : node.inputs) {
            auto it = live_tensors.find(input_name);
            if (it == live_tensors.end()) {
                throw std::runtime_error("Missing input tensor at execution time: " + input_name);
            }
            inputs.push_back(it->second);
        }

        const auto output_shape = node.op->inferOutputShape(inputs);
        Tensor output(output_shape, pool.acquire(Tensor::computeNumel(output_shape)), true);

        const auto op_start = std::chrono::high_resolution_clock::now();
        node.op->execute(inputs, output);
        const auto op_end = std::chrono::high_resolution_clock::now();

        result.operator_profiles.push_back(OperatorProfile{
            node.name,
            node.op->type(),
            std::chrono::duration<double, std::milli>(op_end - op_start).count()
        });

        live_tensors[node.output] = output;

        if (options_.enable_buffer_reuse) {
            for (const auto& input_name : node.inputs) {
                auto release_time = last_use.find(input_name);
                if (release_time != last_use.end() &&
                    release_time->second == position &&
                    !graph.isInitialValue(input_name) &&
                    !graph.isGraphOutput(input_name)) {
                    auto tensor_it = live_tensors.find(input_name);
                    if (tensor_it != live_tensors.end() && tensor_it->second.poolManaged()) {
                        pool.release(tensor_it->second.buffer());
                    }
                    live_tensors.erase(input_name);
                }
            }
        }
    }

    const auto total_end = std::chrono::high_resolution_clock::now();
    result.total_latency_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    result.memory_stats = pool.stats();

    for (const auto& output_name : graph.outputs()) {
        auto it = live_tensors.find(output_name);
        if (it == live_tensors.end()) {
            throw std::runtime_error("Requested graph output not produced: " + output_name);
        }
        result.outputs[output_name] = it->second;
    }

    return result;
}

}  // namespace runtime
