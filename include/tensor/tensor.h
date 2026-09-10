#pragma once

#include "base/base.h"
#include "base/buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tensor {

class Tensor {
  public:
    Tensor(
        std::vector<std::int64_t> dims,
        base::DataType data_type,
        std::shared_ptr<base::DeviceAllocator> allocator);
    ~Tensor() = default;

    Tensor(const Tensor& other) = delete;
    Tensor& operator=(const Tensor& other) = delete;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // offset 单位是元素，相对于当前 Tensor，不是相对于整个 Buffer。
    Tensor view(std::vector<int64_t> dims, std::size_t offset = 0);
    bool overlaps(const Tensor& other) const;

    bool empty() const;
    std::size_t dims_size() const;
    std::int64_t dim(size_t index) const;
    bool same_shape(const Tensor& other) const;
    std::size_t size() const;
    std::size_t byte_size() const;
    const std::vector<std::int64_t>& dims() const;
    base::DataType data_type() const;
    base::DeviceType device_type() const;

    template <typename T>
    T* ptr() noexcept {
        if (empty())
            return nullptr;
        std::byte* start = static_cast<std::byte*>(buffer_->ptr());
        return reinterpret_cast<T*>(start + byte_offset_);
    }
    template <typename T>
    const T* ptr() const noexcept {
        if (empty())
            return nullptr;
        const std::byte* start = static_cast<const std::byte*>(buffer_->ptr());
        return reinterpret_cast<const T*>(start + byte_offset_);
    }

  private:
    Tensor(
        std::vector<std::int64_t> dims,
        base::DataType data_type,
        std::shared_ptr<base::Buffer> buffer,
        std::size_t byte_offset);

    // 声明顺序很重要，构造顺序永远按照这里的顺序
    std::vector<int64_t> dims_; // Tensor 维度
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;
    size_t size_ = 0; // Tensor 元素数量
    std::shared_ptr<base::Buffer> buffer_;
    size_t byte_offset_ = 0;
};

} // namespace tensor
