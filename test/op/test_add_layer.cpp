#include "base/alloc.h"
#include "op/add.h"
#include "tensor/tensor.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

TEST(AddLayerTest, AddsTwoCpuFp32Tensors)
{
    auto allocator = std::make_shared<base::CPUDeviceAllocator>();

    tensor::Tensor left({4}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor right({4}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor output({4}, base::DataType::kDataTypeFp32, allocator);

    float *left_data = left.ptr<float>();
    float *right_data = right.ptr<float>();

    ASSERT_NE(left_data, nullptr);
    ASSERT_NE(right_data, nullptr);

    left_data[0] = 1.0F;
    left_data[1] = 2.0F;
    left_data[2] = 3.0F;
    left_data[3] = 4.0F;

    right_data[0] = 10.0F;
    right_data[1] = 20.0F;
    right_data[2] = 30.0F;
    right_data[3] = 40.0F;

    op::AddLayer layer;

    base::Status status = layer.forward(
        {&left, &right},
        {&output});

    ASSERT_TRUE(status) << status.get_err_message();

    const float *result = output.ptr<float>();

    ASSERT_NE(result, nullptr);

    EXPECT_FLOAT_EQ(result[0], 11.0F);
    EXPECT_FLOAT_EQ(result[1], 22.0F);
    EXPECT_FLOAT_EQ(result[2], 33.0F);
    EXPECT_FLOAT_EQ(result[3], 44.0F);
}

TEST(AddLayerTest, RejectsWrongOutputShape) {
    auto allocator = std::make_shared<base::CPUDeviceAllocator>();

    tensor::Tensor left({4}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor right({4}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor wrong_output({2}, base::DataType::kDataTypeFp32, allocator);

    op::AddLayer layer;

    base::Status status = layer.forward(
        {&left, &right},
        {&wrong_output}
    );

    EXPECT_FALSE(status);
    EXPECT_EQ(
        status.get_err_code(),
        base::kInvalidArgument
    );
}

TEST(AddLayerTest, RejectsNullTensor) {
    auto allocator = std::make_shared<base::CPUDeviceAllocator>();

    tensor::Tensor input({4}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor output({4}, base::DataType::kDataTypeFp32, allocator);


    op::AddLayer layer;

    base::Status status = layer.forward(
        {&input, nullptr},
        {&output}
    );

    EXPECT_FALSE(status);
    EXPECT_EQ(
        status.get_err_code(),
        base::kInvalidArgument
    );
}