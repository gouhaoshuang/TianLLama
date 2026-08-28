#pragma once
#include<cstddef>
#include <cstdlib>


enum class DeviceType{
    kDeviceUnknown = 0,
    kDeviceCPU,
    kDeviceGPU
};


class DeviceAllocator{

public:
    // 构造函数，指定设备类型
    explicit DeviceAllocator(DeviceType device_type) : device_type_(device_type){}
    virtual ~DeviceAllocator() = default;

    DeviceType device_type() const{
        return device_type_;
    }

    
    virtual void* allocate(std::size_t byte_size) const = 0;
    virtual void release(void* ptr) const = 0;


private:
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
};


class CPUDeviceAllocator final: public DeviceAllocator
{
public:
    CPUDeviceAllocator();

    void* allocate(std::size_t byte_size) const override;
    void release(void* ptr) const override;
};

// class CUDADeviceAllocator : public DeviceAllocator
// {

// public:
//     CUDADeviceAllocator();


//     void* allocate(std::size_t byte_size) const override;
//     void release(void* ptr) const override;
// };