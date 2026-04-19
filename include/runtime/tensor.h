#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace runtime {

struct Buffer {
    explicit Buffer(std::size_t capacity_elements)
        : data(capacity_elements, 0.0f) {}

    std::size_t capacity() const { return data.size(); }
    std::size_t bytes() const { return data.size() * sizeof(float); }

    std::vector<float> data;
};

class Tensor {
public:
    Tensor() = default;

    Tensor(std::vector<int> shape, std::shared_ptr<Buffer> buffer, bool pool_managed = false)
        : shape_(std::move(shape)), buffer_(std::move(buffer)), pool_managed_(pool_managed) {
        if (!buffer_) {
            throw std::invalid_argument("Tensor requires a valid buffer.");
        }
        if (buffer_->capacity() < numel()) {
            throw std::invalid_argument("Buffer capacity is smaller than tensor size.");
        }
    }

    static Tensor fromVector(const std::vector<int>& shape, const std::vector<float>& values) {
        auto buffer = std::make_shared<Buffer>(values.size());
        buffer->data = values;
        return Tensor(shape, std::move(buffer), false);
    }

    static Tensor zeros(const std::vector<int>& shape) {
        auto elements = computeNumel(shape);
        auto buffer = std::make_shared<Buffer>(elements);
        std::fill(buffer->data.begin(), buffer->data.end(), 0.0f);
        return Tensor(shape, std::move(buffer), false);
    }

    const std::vector<int>& shape() const { return shape_; }

    void setShape(const std::vector<int>& new_shape) {
        if (!buffer_ || buffer_->capacity() < computeNumel(new_shape)) {
            throw std::invalid_argument("New shape exceeds tensor buffer capacity.");
        }
        shape_ = new_shape;
    }

    std::size_t rank() const { return shape_.size(); }

    std::size_t numel() const { return computeNumel(shape_); }

    float* data() { return buffer_->data.data(); }
    const float* data() const { return buffer_->data.data(); }

    float& operator[](std::size_t index) { return buffer_->data[index]; }
    const float& operator[](std::size_t index) const { return buffer_->data[index]; }

    const std::shared_ptr<Buffer>& buffer() const { return buffer_; }
    bool poolManaged() const { return pool_managed_; }

    static std::size_t computeNumel(const std::vector<int>& shape) {
        if (shape.empty()) {
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
            [](std::size_t acc, int dim) {
                if (dim <= 0) {
                    throw std::invalid_argument("Tensor dimensions must be positive.");
                }
                return acc * static_cast<std::size_t>(dim);
            });
    }

private:
    std::vector<int> shape_;
    std::shared_ptr<Buffer> buffer_;
    bool pool_managed_ = false;
};

}  // namespace runtime
