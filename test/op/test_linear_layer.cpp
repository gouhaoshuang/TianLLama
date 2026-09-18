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
    const std::vector<float>& values) {
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
std::shared_ptr<tensor::Tensor> make_linear_data(
    const std::vector<int64_t>& shape,
    const std::vector<float>& values,
    const std::shared_ptr<base::DeviceAllocator>& allocator) {
    auto t = std::make_shared<tensor::Tensor>(
        shape,
        base::DataType::kDataTypeFp32,
        allocator);
    if (t->size() != values.size())
        throw std::invalid_argument("Linear test data size mismatch");
    if (allocator->device_type() == base::DeviceType::kDeviceCPU)
        std::copy(values.begin(), values.end(), t->ptr<float>());
    else
        allocator->memcpy(values.data(), t->ptr<float>(), t->byte_size(),
                          base::MemcpyKind::kMemcpyCPU2GPU); // 默认流同步上传
    return t;
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
    const op::Layer& base_layer = layer; // 验证统一多态接口。

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

TEST(LinearLayerTest, CpuOptionalBiasAndValidation) {
    auto cpu = std::make_shared<base::CPUDeviceAllocator>();
    auto w = make_linear_data({3,2}, {1,0, 0,1, 1,1}, cpu);
    auto b = make_linear_data({3}, {0.5F,-1,2}, cpu);
    auto x = make_linear_data({2,2}, {1,2, 3,4}, cpu);
    tensor::Tensor y({2,3}, base::DataType::kDataTypeFp32, cpu);
    const std::vector<float> plain{1,2,3, 3,4,7};
    const std::vector<float> biased{1.5F,1,5, 3.5F,3,9};

    op::LinearLayer no_bias(w);
    auto status = no_bias.forward({x.get()}, {&y});
    ASSERT_TRUE(status) << status.get_err_message();
    for (std::size_t i = 0; i < plain.size(); ++i)
        EXPECT_FLOAT_EQ(y.ptr<float>()[i], plain[i]);

    op::LinearLayer with_bias(w, b);
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::fill_n(y.ptr<float>(), y.size(), 123.0F);
        status = with_bias.forward({x.get()}, {&y});
        ASSERT_TRUE(status) << status.get_err_message();
        for (std::size_t i = 0; i < biased.size(); ++i)
            EXPECT_FLOAT_EQ(y.ptr<float>()[i], biased[i]);
    }

    auto wrong_length = make_linear_data({2}, {0,0}, cpu);
    auto wrong_rank = make_linear_data({1,3}, {0,0,0}, cpu);
    EXPECT_THROW((void)op::LinearLayer(w, wrong_length), std::invalid_argument);
    EXPECT_THROW((void)op::LinearLayer(w, wrong_rank), std::invalid_argument);

    // 用同一份 bias 存储构造 [1,N] 输出，验证 overlap，而不触发 shape 错误。
    auto x_one = x->view({1,2});
    auto alias_output = b->view({1,3});
    status = with_bias.forward({&x_one}, {&alias_output});
    EXPECT_EQ(status.get_err_code(), base::kInvalidArgument);
    EXPECT_FLOAT_EQ(b->ptr<float>()[0], 0.5F);
    EXPECT_FLOAT_EQ(b->ptr<float>()[1], -1.0F);
    EXPECT_FLOAT_EQ(b->ptr<float>()[2], 2.0F);
}