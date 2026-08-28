
#include "base/alloc.h"
#include<cstdlib>


CPUDeviceAllocator::CPUDeviceAllocator(): DeviceAllocator(DeviceType::kDeviceCPU){}

void* CPUDeviceAllocator::allocate(std::size_t  byte_size) const  {
    if ( byte_size == 0 ){
        return nullptr;
    }
    return std::malloc(byte_size);
}

void CPUDeviceAllocator::release(void* ptr) const{
    std::free(ptr);
} 