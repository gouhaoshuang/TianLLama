#include "base/alloc.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <exception>
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

    void * ptr = allocator.allocate(64);

    if (ptr == nullptr){
        std::cerr << "错误：内存分配失败\n";
        return false;
    }

    allocator.memset_zero(ptr , 64);
    auto* bytes = static_cast<unsigned char*>(ptr);

    for(std::size_t i = 0 ; i < 64 ; i ++){
        if(bytes[i] != 0){
            std::cerr << "CPU memset_zero test failed\n";
            allocator.release(ptr);
            return false;
        }
    }
    allocator.release(ptr);

    std::array<int , 4> source{1 , 2, 3, 4};
    std::array<int,4> destination{};

    allocator.memcpy(
        source.data(),
        destination.data(),
        sizeof(source),
        MemcpyKind::kMemcpyCPU2CPU
    );

    if(source != destination){
        std::cerr << "CPU memcpy test failed! \n";
        return false;
    }

    std::cout << "CPU allocator test passed\n";
    return true;
}

bool test_gpu_allocator(){
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if(status != cudaSuccess || device_count == 0){
        std::cout << "CUDA test skipped: no CUDA device\n";
        return true;
    }

    CUDADeviceAllocator allocator;
    if(allocator.device_type() != DeviceType::kDeviceGPU){
        std::cerr << "GPU device type test failed \n";
        return false;
    }
    if(allocator.allocate(0)  != nullptr){
        std::cerr << "GPU allocate(0) test failed \n";
        return false;
    }
    std::array<int, 4> source{10, 20, 30, 40};
    std::array<int, 4> destination{};

    void * gpu_source  = allocator.allocate(sizeof(source));
    void * gpu_destination  = allocator.allocate(sizeof(source));

    if (gpu_source == nullptr || gpu_destination == nullptr) {
        std::cerr << "GPU allocation failed\n";

        allocator.release(gpu_source);
        allocator.release(gpu_destination);
        return false;
    }
    
    allocator.memcpy(
        source.data(),
        gpu_source,
        sizeof(source),
        MemcpyKind::kMemcpyCPU2GPU
    );

    allocator.memcpy(
        gpu_source,
        gpu_destination,
        sizeof(source),
        MemcpyKind::kMemcpyGPU2GPU
    );

    allocator.memcpy(
        gpu_destination,
        destination.data(),
        sizeof(destination),
        MemcpyKind::kMemcpyGPU2CPU
    );

    if (source != destination) {
        std::cerr << "GPU memcpy test failed\n";

        allocator.release(gpu_source);
        allocator.release(gpu_destination);
        return false;
    }


    allocator.memset_zero(
        gpu_destination,
        sizeof(source),
        nullptr,
        true
    );
   allocator.memcpy(
        gpu_destination,
        destination.data(),
        sizeof(destination),
        MemcpyKind::kMemcpyGPU2CPU
    );

    bool all_zero = std::all_of(
        destination.begin(),
        destination.end(),
        [](int value) {
            return value == 0;
        }
    );

    allocator.release(gpu_source);
    allocator.release(gpu_destination);

    if (!all_zero) {
        std::cerr << "GPU memset_zero test failed\n";
        return false;
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