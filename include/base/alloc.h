#pragma once
#include<cstddef>
#include "base.h"


namespace base
{

class DeviceAllocator{

public:
    // 构造函数，指定设备类型
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type){}
    virtual ~DeviceAllocator() noexcept = default;

    DeviceType device_type() const{
        return device_type_;
    }
    
    virtual void* allocate(std::size_t byte_size) const = 0;
    virtual void release(void* ptr) const noexcept = 0;

        // 内存、显存拷贝
    virtual void memcpy(
        const void* src_ptr, void* dest_ptr, std::size_t byte_size,
        MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,
        void* stream = nullptr, bool need_sync = false
    ) const;

    // // 内存、 显存清零
    virtual void memset_zero(
        void* ptr , size_t byte_size, 
        void* stream = nullptr , bool need_sync = false
    ) const;



private:
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
};


class CPUDeviceAllocator final: public DeviceAllocator
{
public:
    CPUDeviceAllocator();

    void* allocate(std::size_t byte_size) const override;
    void release(void* ptr) const noexcept override;
};

class CUDADeviceAllocator : public DeviceAllocator
{

public:
    CUDADeviceAllocator();


    void* allocate(std::size_t byte_size) const override;
    void release(void* ptr) const noexcept override;
};


} // namespace base
