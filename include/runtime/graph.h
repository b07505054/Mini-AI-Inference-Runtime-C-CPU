#pragma once

#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "runtime/operator.h"
#include "runtime/tensor.h"

namespace runtime {

struct Node {
    std::string name;
    OperatorPtr op;
    std::vector<std::string> inputs;
    std::string output;
};

class Graph {
public:
    void addNode(Node node) {
        if (!node.op) {
            throw std::invalid_argument("Graph node requires an operator.");
        }
        nodes_.push_back(std::move(node));
    }

    void addValue(const std::string& name, const Tensor& tensor) {
        initial_values_[name] = tensor;
    }

    void setOutputs(std::vector<std::string> outputs) {
        outputs_ = std::move(outputs);
    }

    const std::vector<Node>& nodes() const { return nodes_; }
    const std::unordered_map<std::string, Tensor>& initialValues() const { return initial_values_; }
    const std::vector<std::string>& outputs() const { return outputs_; }

    bool isInitialValue(const std::string& name) const {
        return initial_values_.count(name) > 0;
    }

    bool isGraphOutput(const std::string& name) const {
        return std::find(outputs_.begin(), outputs_.end(), name) != outputs_.end();
    }

    std::unordered_map<std::string, int> producerMap() const {
        std::unordered_map<std::string, int> producers;
        for (int i = 0; i < static_cast<int>(nodes_.size()); ++i) {
            if (producers.count(nodes_[i].output) != 0) {
                throw std::runtime_error("Duplicate output tensor name detected: " + nodes_[i].output);
            }
            producers[nodes_[i].output] = i;
        }
        return producers;
    }

    std::vector<int> topologicalOrder() const;

private:
    std::vector<Node> nodes_;
    std::unordered_map<std::string, Tensor> initial_values_;
    std::vector<std::string> outputs_;
};

}  // namespace runtime
