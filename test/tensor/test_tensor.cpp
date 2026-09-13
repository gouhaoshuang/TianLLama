#include "base/buffer.h"
#include "tensor/tensor.h"
#include <cuda_runtime_api.h>


#include<gtest/gtest.h>
#include <cassert>
#include <iostream>
#include <memory>

TEST(TensorTest, tensor_test){
  auto allocator = std::make_shared<base::CPUDeviceAllocator>();

    tensor::Tensor value(
        {2, 3},
        base::DataType::kDataTypeFp32,
        allocator);



    EXPECT_TRUE(value.size() == 6);
    EXPECT_TRUE(value.byte_size() == 6 * sizeof(float));
    EXPECT_TRUE(value.device_type() == base::DeviceType::kDeviceCPU);

    EXPECT_TRUE(value.dims().size() == 2);
    EXPECT_TRUE(value.dims()[0] == 2);
    EXPECT_TRUE(value.dims()[1] == 3);

    float *data = value.ptr<float>();
    EXPECT_TRUE(data != nullptr);

    data[0] = 1.25F;

    const tensor::Tensor &read_only_value = value;
    const float *read_only_data = read_only_value.ptr<float>();

    EXPECT_TRUE(read_only_data[0] == 1.25F);

    EXPECT_FALSE(value.empty());
    EXPECT_FALSE(value.dims_size() != 2);
    EXPECT_FALSE(value.dim(0) != 2);



    tensor::Tensor same(
        {2, 3},
        base::DataType::kDataTypeFp32,
        allocator);

    tensor::Tensor different(
        {3, 2},
        base::DataType::kDataTypeFp32,
        allocator);

    if (!value.same_shape(same) ||
        value.same_shape(different)) {
    }

}


