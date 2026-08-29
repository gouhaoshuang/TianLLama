
#include "base/alloc.h"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>



namespace{
void check_cuda(cudaError_t status , const char * operation){
    if (status == cudaSuccess){
        return ;
    }

    throw std::runtime_error(
        std::string(operation) + " 失败：" +
        cudaGetErrorString(status)
    );
}
}

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceGPU){}

void* CUDADeviceAllocator:: allocate(std::size_t  byte_size) const {
    if( byte_size == 0){
        return nullptr;
    }

    void * ptr = nullptr;

    check_cuda(cudaMalloc(&ptr , byte_size) , "cudaMalloc");
    return ptr;
}

void CUDADeviceAllocator::release(void* ptr) const{
    if (ptr == nullptr){
        return;
    }
    check_cuda(cudaFree(ptr) , "cudaFree");
}

