#pragma once

#include"base/base.h"
#include"base/buffer.h"

#include<cstddef>
#include <cstdint>
#include <vector>
#include<memory>

namespace tensor
{
    

class Tensor
{
public:

    Tensor(
        std::vector<std::int64_t> dims,
        base::DataType data_type,
        std::shared_ptr<base::DeviceAllocator> allocator
    );
    ~Tensor() = default;


    Tensor(const Tensor& other) = delete;
    Tensor& operator=(const Tensor& other) = delete;

    Tensor( Tensor&& other) noexcept = default;
    Tensor& operator=( Tensor&& other) = default;

    std::size_t size() const;
    std::size_t byte_size() const;

    const std::vector<std::int64_t>& dims() const;

    base::DataType data_type() const;
    base::DeviceType device_type() const;


    template<typename T>
    T* ptr() noexcept{
        return static_cast<T*>(buffer_.ptr());
    }
    template<typename T>
    const T* ptr() const noexcept{
        return static_cast<const T*>(buffer_.ptr());
    }


private:
    // 声明顺序很重要，构造顺序永远按照这里的顺序
    std::vector<int64_t> dims_; // Tensor 维度
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;
    size_t size_ = 0; // Tensor 元素数量
    base::Buffer buffer_;
};


} // namespace tensor
