#include "base/base.h"
#include "op/linear.h"
#include "tensor/tensor.h"
#include <gtest/gtest.h>

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>

namespace {

std::shared_ptr<tensor::Tensor> make_weight(
    int64_t N,
    int64_t K,
    const std::vector<float> &values) {
    std::shared_ptr<base::CPUDeviceAllocator> allocator = std::make_shared<base::CPUDeviceAllocator>();

    std::shared_ptr<tensor::Tensor> weight = std::make_shared<tensor::Tensor>(
        std::vector<std::int64_t>{N, K},
        base::DataType::kDataTypeFp32,
        allocator);

    if (values.size() != weight->size()) {
        throw std::invalid_argument("Test weight size mismatch");
    }
    std::copy(values.begin(), values.end(), weight->ptr<float>());
    return weight;
}

} // namespace

TEST(LinearLayerTest, CpuFixedNonSquareAndRepeat) {
    std::shared_ptr<base::CPUDeviceAllocator> allocator = std::make_shared<base::CPUDeviceAllocator>();
    // W:[4 , 3]
    auto w = make_weight(4, 3, {1, 0, 2, 0, 1, -1, 1, 1, 1, -2, 1, 0});

    tensor::Tensor x({2, 3}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor y({2, 4}, base::DataType::kDataTypeFp32, allocator);

    const std::vector<float> input{1, 2, 3, -1, 0, 2};
    const std::vector<float> expected{7, -1, 6, 0, 3, -2, 1, 2};

    std::copy(input.begin(), input.end(), x.ptr<float>());

    op::LinearLayer layer(w);
    const op::Layer &base_layer = layer; // 验证统一多态接口。

    for (int repeat = 0; repeat < 2; ++repeat) {

        std::fill_n(y.ptr<float>(), y.size(), 123.0F);

        auto status = base_layer.forward({&x}, {&y});

        ASSERT_EQ(status.get_err_code(), base::kSuccess) << status.get_err_message();
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_FLOAT_EQ(y.ptr<float>()[i], expected[i]);
        }
    }
}

TEST(LinearLayerTest, CpuSingleToken) {
    auto allocator = std::make_shared<base::CPUDeviceAllocator>();

    auto w = make_weight(2, 3, {1, 2, 3, 4, 5, 6});

    tensor::Tensor x({1, 3}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor y({1, 2}, base::DataType::kDataTypeFp32, allocator);

    const std::vector<float> input{7, 8, 9};
    std::copy(input.begin(), input.end(), x.ptr<float>());

    op::LinearLayer layer(w);
    auto status = layer.forward({&x}, {&y});

    ASSERT_EQ(status.get_err_code(), base::kSuccess);

    EXPECT_FLOAT_EQ(y.ptr<float>()[0], 50.0F);
    EXPECT_FLOAT_EQ(y.ptr<float>()[1], 122.0F);
}

TEST(LinearLayerTest, CudaFixedNonSquare) {
    int device_count = 0;
    const auto device_status = cudaGetDeviceCount(&device_count);
    if (device_status == cudaErrorNoDevice ||
        (device_status == cudaSuccess && device_count == 0)) {
        GTEST_SKIP() << "No CUDA device";
    }
    ASSERT_EQ(device_status, cudaSuccess) << cudaGetErrorString(device_status);

    auto cpu = std::make_shared<base::CPUDeviceAllocator>();
    auto gpu = std::make_shared<base::CUDADeviceAllocator>();
    const std::vector<float> input{1, 2, 3, -1, 0, 2};
    const std::vector<float> weights{1, 0, 2, 0, 1, -1, 1, 1, 1, -2, 1, 0};
    const std::vector<float> expected{7, -1, 6, 0, 3, -2, 1, 2};

    // 1. 用固定权重在 CPU 上计算。
    auto w_cpu = make_weight(4, 3, weights);

    tensor::Tensor x_cpu({2, 3}, base::DataType::kDataTypeFp32, cpu);
    tensor::Tensor y_cpu({2, 4}, base::DataType::kDataTypeFp32, cpu);
    std::copy(input.begin(), input.end(), x_cpu.ptr<float>());

    op::LinearLayer cpu_layer(w_cpu);
    const auto cpu_status = cpu_layer.forward({&x_cpu}, {&y_cpu});
    ASSERT_TRUE(cpu_status) << cpu_status.get_err_message();

    // 2. 在 GPU 上分配相同形状的 Tensor，上传同一份输入和权重。
    auto w_gpu = std::make_shared<tensor::Tensor>(
        std::vector<std::int64_t>{4, 3},
        base::DataType::kDataTypeFp32,
        gpu);
    tensor::Tensor x_gpu({2, 3}, base::DataType::kDataTypeFp32, gpu);
    tensor::Tensor y_gpu({2, 4}, base::DataType::kDataTypeFp32, gpu);

    gpu->memcpy(
        x_cpu.ptr<float>(),
        x_gpu.ptr<float>(),
        x_cpu.byte_size(),
        base::MemcpyKind::kMemcpyCPU2GPU);
    gpu->memcpy(
        w_cpu->ptr<float>(),
        w_gpu->ptr<float>(),
        w_cpu->byte_size(),
        base::MemcpyKind::kMemcpyCPU2GPU);

    // 3. GPU Layer 使用 GPU 权重，默认流即可。
    op::LinearLayer gpu_layer(w_gpu);
    const auto gpu_status = gpu_layer.forward({&x_gpu}, {&y_gpu});
    ASSERT_TRUE(gpu_status) << gpu_status.get_err_message();

    // 4. 下载结果：不能在 CPU 上直接读取 y_gpu.ptr<float>()[i]。
    std::vector<float> actual(expected.size());
    gpu->memcpy(
        y_gpu.ptr<float>(),
        actual.data(),
        y_gpu.byte_size(),
        base::MemcpyKind::kMemcpyGPU2CPU);

    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(y_cpu.ptr<float>()[i], expected[i], 1e-5F) << "CPU index=" << i;
        EXPECT_NEAR(actual[i], expected[i], 1e-5F) << "GPU index=" << i;
        EXPECT_NEAR(actual[i], y_cpu.ptr<float>()[i], 1e-5F) << "CPU/GPU index=" << i;
    }
}