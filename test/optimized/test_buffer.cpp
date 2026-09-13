#include "base/buffer.h"

#include <cuda_runtime_api.h>

#include<gtest/gtest.h>
#include<memory>
#include<cassert>
#include <iostream>


TEST(BufferTest, test_buffer){
    auto allocator = std::make_shared<base::CPUDeviceAllocator>();


    {
        base::Buffer buffer(32, allocator);
        EXPECT_TRUE(buffer.ptr() != nullptr);
        EXPECT_TRUE(buffer.byte_size() == 32);
    }

    float* external_ptr = new float[32];

    {
        base::Buffer buffer(
            32 * sizeof(float),
            external_ptr,
            base::DeviceType::kDeviceCPU
        );

        EXPECT_TRUE(buffer.is_external());
        EXPECT_TRUE(buffer.ptr() == external_ptr);
        
    }
    external_ptr[0] = 1.0F;
    delete[] external_ptr;
}
