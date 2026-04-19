#include "runtime/graph.h"

namespace runtime {

std::vector<int> Graph::topologicalOrder() const {
    auto producers = producerMap();

    std::vector<std::vector<int>> adjacency(nodes_.size());
    std::vector<int> indegree(nodes_.size(), 0);

    for (int consumer = 0; consumer < static_cast<int>(nodes_.size()); ++consumer) {
        for (const auto& input_name : nodes_[consumer].inputs) {
            auto it = producers.find(input_name);
            if (it != producers.end()) {
                adjacency[it->second].push_back(consumer);
                ++indegree[consumer];
            } else if (!isInitialValue(input_name)) {
                throw std::runtime_error("Input tensor not found in graph: " + input_name);
            }
        }
    }

    std::queue<int> ready;
    for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
        if (indegree[i] == 0) {
            ready.push(i);
        }
    }

    std::vector<int> order;
    while (!ready.empty()) {
        int node_index = ready.front();
        ready.pop();
        order.push_back(node_index);

        for (int next : adjacency[node_index]) {
            --indegree[next];
            if (indegree[next] == 0) {
                ready.push(next);
            }
        }
    }

    if (order.size() != nodes_.size()) {
        throw std::runtime_error("Graph contains a cycle.");
    }

    return order;
}

}  // namespace runtime
