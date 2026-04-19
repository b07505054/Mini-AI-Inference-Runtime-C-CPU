#include "runtime/operators/matmul.h"

#include <stdexcept>

namespace runtime {

std::vector<int> MatMul::inferOutputShape(const std::vector<Tensor>& inputs) const {
    if (inputs.size() != 2) {
        throw std::invalid_argument("MatMul expects exactly 2 inputs.");
    }
    if (inputs[0].rank() != 2 || inputs[1].rank() != 2) {
        throw std::invalid_argument("MatMul expects rank-2 tensors.");
    }
    const auto& a_shape = inputs[0].shape();
    const auto& b_shape = inputs[1].shape();
    if (a_shape[1] != b_shape[0]) {
        throw std::invalid_argument("MatMul dimension mismatch.");
    }
    return {a_shape[0], b_shape[1]};
}

void MatMul::execute(const std::vector<Tensor>& inputs, Tensor& output) const {
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    const int m = a.shape()[0];
    const int k = a.shape()[1];
    const int n = b.shape()[1];

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                sum += a.data()[row * k + inner] * b.data()[inner * n + col];
            }
            output.data()[row * n + col] = sum;
        }
    }
}

}  // namespace runtime
