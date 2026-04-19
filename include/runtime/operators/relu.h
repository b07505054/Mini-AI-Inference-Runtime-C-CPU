#pragma once

#include "runtime/operator.h"

namespace runtime {

class ReLU final : public Operator {
public:
    std::string type() const override { return "ReLU"; }
    std::vector<int> inferOutputShape(const std::vector<Tensor>& inputs) const override;
    void execute(const std::vector<Tensor>& inputs, Tensor& output) const override;
};

}  // namespace runtime
