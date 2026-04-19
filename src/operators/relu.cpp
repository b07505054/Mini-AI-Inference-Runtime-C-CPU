#include "runtime/operators/relu.h"

#include <algorithm>
#include <stdexcept>

namespace runtime {

std::vector<int> ReLU::inferOutputShape(const std::vector<Tensor>& inputs) const {
    if (inputs.size() != 1) {
        throw std::invalid_argument("ReLU expects exactly 1 input.");
    }
    return inputs[0].shape();
}

void ReLU::execute(const std::vector<Tensor>& inputs, Tensor& output) const {
    const auto& input = inputs[0];
    for (std::size_t i = 0; i < input.numel(); ++i) {
        output.data()[i] = std::max(0.0f, input.data()[i]);
    }
}

}  // namespace runtime
