

#include "base/alloc.h"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>
#include <cstring>

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

namespace base
{

void DeviceAllocator::memcpy(
    const void* src_ptr,
    void* dest_ptr,
    std::size_t byte_size,
    MemcpyKind memcpy_kind,
    void* stream,
    bool need_sync
) const {
    if(byte_size == 0) return;

    if(src_ptr == nullptr || dest_ptr == nullptr){
        throw std::invalid_argument(
            "DeviceAllocator::memcpy 接收到了空指针"
        );
    }

    if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU){
        std::memcpy(dest_ptr , src_ptr , byte_size);
        return;
    }

    cudaMemcpyKind cuda_kind;
    switch(memcpy_kind){
        case MemcpyKind::kMemcpyCPU2GPU :
            cuda_kind = cudaMemcpyHostToDevice;
            break;
        case MemcpyKind::kMemcpyGPU2CPU :
            cuda_kind = cudaMemcpyDeviceToHost;
            break;
        case MemcpyKind::kMemcpyGPU2GPU :
            cuda_kind = cudaMemcpyDeviceToDevice;
            break;
        default:
            throw std::invalid_argument("不知道的类型：MemcpyKind！");
    }

    cudaStream_t cuda_stream =  static_cast<cudaStream_t>(stream);
    if(cuda_stream == nullptr){
        check_cuda(
            cudaMemcpy(dest_ptr , src_ptr , byte_size , cuda_kind),
            "cudaMemcpy"
        );
    }else{
        check_cuda(
            cudaMemcpyAsync(
                dest_ptr,
                src_ptr,
                byte_size,
                cuda_kind,
                cuda_stream
            ),
            "cudaMemcpyAsync"
        );
    }
    if (need_sync){
        if(cuda_stream != nullptr){
            check_cuda(
                cudaStreamSynchronize(cuda_stream),
                "cudaStreamSynchronize"
            );
        }else{
            check_cuda(
                cudaDeviceSynchronize(),
                "cudaDeviceSynchronize"
            );
        }
    }
}

void DeviceAllocator::memset_zero(
    void* ptr,
    std::size_t byte_size,
    void* stream,
    bool need_sync
) const {

    if(byte_size == 0){
        return ;
    }

    if(ptr == nullptr){
        throw std::invalid_argument(
            "DeviceAllocator::memset_zero 接收到了不合法的参数"
        );
    }

    if(device_type() == DeviceType::kDeviceCPU){
        std::memset(ptr , 0 , byte_size);
        return;
    }

    if(device_type() != DeviceType::kDeviceGPU){
        throw std::logic_error(
            "DeviceAllocator::memset_zero 不能清除未知设备的内存"
        );
    }

    cudaStream_t cuda_stream =  static_cast<cudaStream_t>(stream);

    if(cuda_stream == nullptr){
        check_cuda(
            cudaMemset(ptr , 0 , byte_size),
            "cudaMemset"
        );
    }else{
        check_cuda(
            cudaMemsetAsync(ptr , 0 , byte_size , cuda_stream),
            "cudaMemsetAsync"
        );
    }
    if (need_sync){
        if(cuda_stream != nullptr){
            check_cuda(
                cudaStreamSynchronize(cuda_stream),
                "cudaStreamSynchronize"
            );
        }else{
            check_cuda(
                cudaDeviceSynchronize(),
                "cudaDeviceSynchronize"
            );
        }
    }

}
} // namespace base
