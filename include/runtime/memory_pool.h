#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <vector>

#include "runtime/tensor.h"

namespace runtime {

struct MemoryStats {
    std::size_t new_allocations = 0;
    std::size_t buffer_reuses = 0;
    std::size_t peak_active_bytes = 0;
    std::size_t peak_reserved_bytes = 0;
    std::size_t current_active_bytes = 0;
    std::size_t current_reserved_bytes = 0;
};

class MemoryPool {
public:
    explicit MemoryPool(bool enable_reuse) : enable_reuse_(enable_reuse) {}

    std::shared_ptr<Buffer> acquire(std::size_t elements) {
        std::shared_ptr<Buffer> selected;
        if (enable_reuse_) {
            std::size_t best_index = std::numeric_limits<std::size_t>::max();
            std::size_t best_capacity = std::numeric_limits<std::size_t>::max();
            for (std::size_t i = 0; i < free_list_.size(); ++i) {
                auto capacity = free_list_[i]->capacity();
                if (capacity >= elements && capacity < best_capacity) {
                    best_capacity = capacity;
                    best_index = i;
                }
            }
            if (best_index != std::numeric_limits<std::size_t>::max()) {
                selected = free_list_[best_index];
                free_list_.erase(free_list_.begin() + static_cast<long>(best_index));
                ++stats_.buffer_reuses;
            }
        }

        if (!selected) {
            selected = std::make_shared<Buffer>(elements);
            ++stats_.new_allocations;
            stats_.current_reserved_bytes += selected->bytes();
            stats_.peak_reserved_bytes = std::max(stats_.peak_reserved_bytes, stats_.current_reserved_bytes);
        }

        std::fill(selected->data.begin(), selected->data.end(), 0.0f);
        stats_.current_active_bytes += selected->bytes();
        stats_.peak_active_bytes = std::max(stats_.peak_active_bytes, stats_.current_active_bytes);
        return selected;
    }

    void release(const std::shared_ptr<Buffer>& buffer) {
        if (!buffer) {
            return;
        }
        stats_.current_active_bytes -= buffer->bytes();
        if (enable_reuse_) {
            free_list_.push_back(buffer);
        } else {
            stats_.current_reserved_bytes -= buffer->bytes();
        }
    }

    const MemoryStats& stats() const { return stats_; }

private:
    bool enable_reuse_ = false;
    std::vector<std::shared_ptr<Buffer>> free_list_;
    MemoryStats stats_;
};

}  // namespace runtime
