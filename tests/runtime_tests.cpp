#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>

#include "runtime/execution_engine.h"
#include "runtime/graph.h"
#include "runtime/operators/matmul.h"
#include "runtime/operators/relu.h"

int main() {
    runtime::Graph graph;
    graph.addValue("x", runtime::Tensor::fromVector({1, 2}, {1.0f, -2.0f}));
    graph.addValue("w", runtime::Tensor::fromVector({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}));

    auto matmul = std::make_shared<runtime::MatMul>();
    auto relu = std::make_shared<runtime::ReLU>();

    graph.addNode(runtime::Node{"relu", relu, {"mm"}, "out"});
    graph.addNode(runtime::Node{"matmul", matmul, {"x", "w"}, "mm"});
    graph.setOutputs({"out"});

    runtime::ExecutionEngine engine(runtime::ExecutionOptions{true});
    const auto result = engine.run(graph);

    assert(result.execution_order.size() == 2);
    assert(result.execution_order[0] == "matmul");
    assert(result.execution_order[1] == "relu");

    const auto& output = result.outputs.at("out");
    assert(output.shape()[0] == 1 && output.shape()[1] == 2);

    const float expected0 = std::max(0.0f, 1.0f * 1.0f + -2.0f * 3.0f);
    const float expected1 = std::max(0.0f, 1.0f * 2.0f + -2.0f * 4.0f);

    assert(std::fabs(output.data()[0] - expected0) < 1e-5f);
    assert(std::fabs(output.data()[1] - expected1) < 1e-5f);

    std::cout << "All runtime tests passed.\n";
    return 0;
}
