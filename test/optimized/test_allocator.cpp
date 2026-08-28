#include "base/alloc.h"

#include<cstddef>
#include<cstdlib>
#include <iostream>


int main(){
    CPUDeviceAllocator cpu_allocator;
    DeviceAllocator& allocator = cpu_allocator;

    if (allocator.device_type() != DeviceType::kDeviceCPU){
        std::cerr << "错误：设备类型不正确\n";
        return 1;
    }

    if (allocator.allocate(0) != nullptr){
        std::cerr << "错误：分配 0 字节应该返回 nullptr\n";
        return 1;
    }

    size_t byte_size = 64;
    void * ptr = allocator.allocate(byte_size);

    if (ptr == nullptr){
        std::cerr << "错误：内存分配失败\n";
        return 1;
    }

    auto* bytes = static_cast<unsigned char*>(ptr);
    bytes[0] = 1;
    bytes[byte_size - 1] = 2;
    allocator.release(ptr);

    std::cout << "CPU allocator test passed\n";
    return 0;
}