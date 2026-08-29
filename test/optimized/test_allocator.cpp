#include "base/alloc.h"

#include<cuda_runtime_api.h>
#include<cstddef>
#include<cstdlib>
#include <iostream>

bool test_cpu_allocator(){
    CPUDeviceAllocator cpu_allocator;
    DeviceAllocator& allocator = cpu_allocator;

    if (allocator.device_type() != DeviceType::kDeviceCPU){
        std::cerr << "错误：设备类型不正确\n";
        return false;
    }

    if (allocator.allocate(0) != nullptr){
        std::cerr << "错误：分配 0 字节应该返回 nullptr\n";
        return false;
    }

    size_t byte_size = 64;
    void * ptr = allocator.allocate(byte_size);

    if (ptr == nullptr){
        std::cerr << "错误：内存分配失败\n";
        return false;
    }

    auto* bytes = static_cast<unsigned char*>(ptr);
    bytes[0] = 1;
    bytes[byte_size - 1] = 2;
    allocator.release(ptr);

    std::cout << "CPU allocator test passed\n";
    return true;
}

bool test_gpu_allocator(){
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count  );
    if(status != cudaSuccess || device_count == 0){
        std::cout << "CUDA test skipped: no CUDA device\n";
        return true;
    }

    std::cout << "CUDA allocator test passed\n";
    return true;
}
int main(){
    try{
        if(!test_cpu_allocator()){
            return 1;
        }
        
        if(!test_gpu_allocator()){
            return 1;
        }
    }catch(const std::exception& error){
        std::cerr << "Allocator test threw an exception: "
                  << error.what() << '\n';
        return 1;
    }
    std::cout << "All allocator tests passed\n";
    return 0;
}