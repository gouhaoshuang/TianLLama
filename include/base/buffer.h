#pragma once

#include "base/alloc.h"

#include<cstddef>
#include<memory>

namespace base{


class Buffer
{

public:
    Buffer() = default;

    // 拥有型 Buffer：通过 allocator 分配并管理资源
    Buffer(
        std::size_t byte_size,
        std::shared_ptr<DeviceAllocator> allocator
    );

    // 借用型 Buffer：只包装外部指针，不负责释放
    Buffer(
        std::size_t byte_size,
        void* external_ptr,
        DeviceType device_type
    );

    ~Buffer();

    Buffer(const Buffer& other) = delete;  // 删除复制构造函数
    Buffer& operator= (const Buffer& other) = delete; // 删除复制赋值运算符

    Buffer( Buffer&& other) ; // 自定义移动构造函数
    Buffer& operator=( Buffer&& other)  ;  // 自定义移动赋值运算符


    void * ptr();
    const void * ptr() const;

    size_t byte_size() const;
    DeviceType device_type() const;

    bool is_external() const;
    bool owns_memory() const;
    bool empty() const;

    std::shared_ptr<DeviceAllocator> allocator() const;
                  

private:
    void release_owned_memory() noexcept;


private:
    std::size_t byte_size_ = 0;
    void* ptr_ = nullptr;
    bool use_external_ = false;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    std::shared_ptr<DeviceAllocator> allocator_;
};

} // namespace base

