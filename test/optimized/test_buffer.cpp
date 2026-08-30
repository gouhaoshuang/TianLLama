#include "base/buffer.h"

#include <cuda_runtime_api.h>

#include<memory>
#include<cassert>
#include <iostream>


int main(){
    auto allocator = std::make_shared<CPUDeviceAllocator>();


    {
        Buffer buffer(32, allocator);

        assert(buffer.ptr() != nullptr);
        assert(buffer.byte_size() == 32);
    }

    float* external_ptr = new float[32];

    {
        Buffer buffer(
            32 * sizeof(float),
            external_ptr,
            DeviceType::kDeviceCPU
        );

        assert(buffer.is_external());
        assert(buffer.ptr() == external_ptr);
    }

    external_ptr[0] = 1.0F;
    delete[] external_ptr;

    std::cout << "All Buffer tests passed\n";
    return 0;
}