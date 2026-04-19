#pragma once

#include <memory>
#include <string>
#include <vector>

#include "runtime/tensor.h"

namespace runtime {

class Operator {
public:
    virtual ~Operator() = default;

    virtual std::string type() const = 0;
    virtual std::vector<int> inferOutputShape(const std::vector<Tensor>& inputs) const = 0;
    virtual void execute(const std::vector<Tensor>& inputs, Tensor& output) const = 0;
};

using OperatorPtr = std::shared_ptr<Operator>;

}  // namespace runtime
