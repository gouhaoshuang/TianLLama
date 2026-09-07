#include "base/alloc.h"
#include "op/add.h"
#include "tensor/tensor.h"

#include <array>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

TEST(AddLayerTest, AddsTwoCpuFp32Tensors) {
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
        {&wrong_output});

    EXPECT_FALSE(status);
    EXPECT_EQ(
        status.get_err_code(),
        base::kInvalidArgument);
}

TEST(AddLayerTest, RejectsNullTensor) {
    auto allocator = std::make_shared<base::CPUDeviceAllocator>();

    tensor::Tensor input({4}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor output({4}, base::DataType::kDataTypeFp32, allocator);

    op::AddLayer layer;

    base::Status status = layer.forward(
        {&input, nullptr},
        {&output});

    EXPECT_FALSE(status);
    EXPECT_EQ(
        status.get_err_code(),
        base::kInvalidArgument);
}

TEST(AddLayerTest, AddsTwoCudaFp32Tensors) {
    int device_count = 0;
    const cudaError_t cuda_status = cudaGetDeviceCount(&device_count);

    if (cuda_status != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto allocator = std::make_shared<base::CUDADeviceAllocator>();

    tensor::Tensor left({4}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor right({4}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor output({4}, base::DataType::kDataTypeFp32, allocator);

    const std::array<float, 4> host_left{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> host_right{10.0F, 20.0F, 30.0F, 40.0F};
    std::array<float, 4> host_output{};

    allocator->memcpy(
        host_left.data(),
        left.ptr<float>(),
        left.byte_size(),
        base::MemcpyKind::kMemcpyCPU2GPU);

    allocator->memcpy(
        host_right.data(),
        right.ptr<float>(),
        right.byte_size(),
        base::MemcpyKind::kMemcpyCPU2GPU);

    op::AddLayer layer;

    const base::Status status = layer.forward(
        {&left, &right},
        {&output});

    ASSERT_TRUE(status) << status.get_err_message();

    allocator->memcpy(
        output.ptr<float>(),
        host_output.data(),
        output.byte_size(),
        base::MemcpyKind::kMemcpyGPU2CPU);

    EXPECT_FLOAT_EQ(host_output[0], 11.0F);
    EXPECT_FLOAT_EQ(host_output[1], 22.0F);
    EXPECT_FLOAT_EQ(host_output[2], 33.0F);
    EXPECT_FLOAT_EQ(host_output[3], 44.0F);
}

struct TestStream {
    cudaStream_t handle = nullptr;

    TestStream() = default;

    TestStream(const TestStream &other) = delete;
    TestStream &operator=(const TestStream &other) = delete;

    ~TestStream() {
        if (handle != nullptr) {
            cudaStreamDestroy(handle);
        }
    }
};

TEST(AddLayerTest, AddsOnExplicitCudaStream) {
    int device_count = 0;
    const cudaError_t cuda_status = cudaGetDeviceCount(&device_count);

    if (cuda_status != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device is not available";
    }

    auto allocator = std::make_shared<base::CUDADeviceAllocator>();

    tensor::Tensor left({4}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor right({4}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor output({4}, base::DataType::kDataTypeFp32, allocator);

    const std::array<float, 4> host_left{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> host_right{10.0F, 20.0F, 30.0F, 40.0F};
    std::array<float, 4> host_output{};

    TestStream stream;

    ASSERT_EQ(cudaStreamCreate(&stream.handle), cudaSuccess);
    base::ExecutionContext context;
    context.stream = stream.handle;

    allocator->memcpy(
        host_left.data(),
        left.ptr<float>(),
        left.byte_size(),
        base::MemcpyKind::kMemcpyCPU2GPU,
        context.stream,
        true);

    allocator->memcpy(
        host_right.data(),
        right.ptr<float>(),
        right.byte_size(),
        base::MemcpyKind::kMemcpyCPU2GPU,
        context.stream,
        true);

    op::AddLayer layer;

    const base::Status status = layer.forward(
        {&left, &right},
        {&output},
        context);

    ASSERT_TRUE(status) << status.get_err_message();

    allocator->memcpy(
        output.ptr<float>(),
        host_output.data(),
        output.byte_size(),
        base::MemcpyKind::kMemcpyGPU2CPU,
        context.stream,
        true);

    EXPECT_FLOAT_EQ(host_output[0], 11.0F);
    EXPECT_FLOAT_EQ(host_output[1], 22.0F);
    EXPECT_FLOAT_EQ(host_output[2], 33.0F);
    EXPECT_FLOAT_EQ(host_output[3], 44.0F);
}
