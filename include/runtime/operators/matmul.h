#pragma once

#include "runtime/operator.h"

namespace runtime {

class MatMul final : public Operator {
public:
    std::string type() const override { return "MatMul"; }
    std::vector<int> inferOutputShape(const std::vector<Tensor>& inputs) const override;
    void execute(const std::vector<Tensor>& inputs, Tensor& output) const override;
};

}  // namespace runtime
