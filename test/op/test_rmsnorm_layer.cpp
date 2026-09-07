#include "base/base.h"
#include "op/rmsnorm.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <vector>

namespace {
std::shared_ptr<tensor::Tensor> make_cpu_weight(
    const std::vector<float> &values) {

    std::shared_ptr<base::CPUDeviceAllocator> allocator = std::make_shared<base::CPUDeviceAllocator>();

    std::shared_ptr<tensor::Tensor> weight = std::make_shared<tensor::Tensor>(
        std::vector<std::int64_t>{static_cast<std::int64_t>(values.size())},
        base::DataType::kDataTypeFp32,
        allocator);

    std::copy(values.begin(), values.end(), weight->ptr<float>());
    return weight;
}
std::vector<float> reference_rmsnorm(
    const std::vector<float> &x,
    const std::vector<float> &w,
    float epsilon) {

    std::vector<float> result(x.size());
    size_t dim = w.size();

    for (size_t offset = 0; offset < x.size(); offset += dim) {
        double sum = 0.0;

        for (size_t j = 0; j < dim; j++) {
            const double value = x[offset + j];
            sum += value * value;
        }
        const double scale = 1.0 / std::sqrt(sum / dim + epsilon);
        for (std::size_t j = 0; j < dim; j++) {
            result[offset + j] = static_cast<float>(x[offset + j] * scale * w[j]);
        }
    }
    return result;
}

void expect_close(const float *actual, const std::vector<float> &expected) {
    for (std::size_t i = 0; i < expected.size(); i++) {
        const float tolerance = 1e-5F + 1e-4F * std::abs(expected[i]);
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "element  " << i;
    }
}

struct RmsTestStream {
    cudaStream_t handle = nullptr;
    RmsTestStream() = default;
    RmsTestStream(const RmsTestStream &other) = delete;
    RmsTestStream &operator=(const RmsTestStream &other) = delete;

    ~RmsTestStream() {
        if (handle) {
            cudaStreamDestroy(handle);
        }
    }
};

} // namespace

TEST(RmsNormLayerTest, CpuHandComputedVector) {
    std::shared_ptr<base::CPUDeviceAllocator> allocator = std::make_shared<base::CPUDeviceAllocator>();
    std::shared_ptr<tensor::Tensor> weight = make_cpu_weight({2.0F, 0.5F});

    tensor::Tensor input({2}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor output({2}, base::DataType::kDataTypeFp32, allocator);
    input.ptr<float>()[0] = 3.0F;
    input.ptr<float>()[1] = 4.0F;

    op::RmsNormLayer layer(weight, 1e-5F);
    const op::Layer &base_layer = layer;
    base::Status status = base_layer.forward({&input}, {&output});

    ASSERT_EQ(status.get_err_code(), base::kSuccess) << status.get_err_message();

    const float scale = 1.0F / std::sqrt(12.5F + 1e-5F);
    EXPECT_NEAR(output.ptr<float>()[0], 6.0F * scale, 1e-5F);
    EXPECT_NEAR(output.ptr<float>()[1], 2.0F * scale, 1e-5F);
}

TEST(RmsNormLayerTest, CpuNormalizesEachRowAndHandlesZeros) {
    auto allocator = std::make_shared<base::CPUDeviceAllocator>();
    const std::vector<float> x{0, 0, 0, 1, -2, 3};
    const std::vector<float> w{1, 0.5F, -2};
    auto weight = make_cpu_weight(w);
    tensor::Tensor input({2, 3}, base::DataType::kDataTypeFp32, allocator);
    tensor::Tensor output({2, 3}, base::DataType::kDataTypeFp32, allocator);
    std::copy(x.begin(), x.end(), input.ptr<float>());

    op::RmsNormLayer layer(weight, 1e-6F);
    auto status = layer.forward({&input}, {&output});
    ASSERT_EQ(status.get_err_code(), base::kSuccess) << status.get_err_message();

    expect_close(output.ptr<float>(), reference_rmsnorm(x, w, 1e-6F));
    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 0.0F);
}

TEST(RmsNormLayerTest, CudaMatchesCpuAndReferenceOnBothStreams) {
    int device_count = 0;
    cudaError_t query = cudaGetDeviceCount(&device_count);

    if ((device_count == 0 && query == cudaSuccess) || query == cudaErrorNoDevice) {
        GTEST_SKIP() << "No CUDA device";
    }

    ASSERT_EQ(query, cudaSuccess) << cudaGetErrorString(query);

    RmsTestStream custom_stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&custom_stream.handle, cudaStreamNonBlocking), cudaSuccess);

    std::shared_ptr<base::CPUDeviceAllocator> cpu_allocator = std::make_shared<base::CPUDeviceAllocator>();
    std::shared_ptr<base::CUDADeviceAllocator> cuda_allocator = std::make_shared<base::CUDADeviceAllocator>();
    float epsilon = 1e-5F;

    for (bool use_custom : {false, true}) {
        base::ExecutionContext context;
        context.stream = use_custom ? custom_stream.handle : nullptr;

        for (int64_t dim : {1, 2, 31, 128, 129, 257, 1024}) {
            SCOPED_TRACE(::testing::Message() << "D=" << dim << ", custom_stream=" << use_custom);
            const std::vector<std::int64_t> shape{2, 3, dim};
            const size_t d = static_cast<size_t>(dim);
            std::vector<float> host_x(6 * d);
            std::vector<float> host_w(d);

            for (size_t i = 0; i < host_x.size(); i++) {
                host_x[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 5.0F;
            }
            std::fill_n(host_x.begin(), d, 0.0F); // 第一行全零。
            for (std::size_t j = 0; j < d; ++j) {
                host_w[j] = static_cast<float>(static_cast<int>(j % 7) - 3) / 2.0F;
            }
            auto cpu_weight = make_cpu_weight(host_w);
            tensor::Tensor cpu_x(shape, base::DataType::kDataTypeFp32, cpu_allocator);
            tensor::Tensor cpu_y(shape, base::DataType::kDataTypeFp32, cpu_allocator);
            std::copy(host_x.begin(), host_x.end(), cpu_x.ptr<float>());

            op::RmsNormLayer cpu_layer(cpu_weight, epsilon);
            auto cpu_status = cpu_layer.forward({&cpu_x}, {&cpu_y});
            ASSERT_EQ(cpu_status.get_err_code(), base::kSuccess);

            auto gpu_weight = std::make_shared<tensor::Tensor>(
                std::vector<int64_t>{dim},
                base::DataType::kDataTypeFp32,
                cuda_allocator);
            tensor::Tensor gpu_x(shape, base::DataType::kDataTypeFp32, cuda_allocator);
            tensor::Tensor gpu_y(shape, base::DataType::kDataTypeFp32, cuda_allocator);
            std::vector<float> host_y(host_x.size());
            cuda_allocator->memcpy(
                host_x.data(),
                gpu_x.ptr<float>(),
                gpu_x.byte_size(),
                base::MemcpyKind::kMemcpyCPU2GPU,
                context.stream,
                true);
            cuda_allocator->memcpy(
                host_w.data(),
                gpu_weight->ptr<float>(),
                gpu_weight->byte_size(),
                base::MemcpyKind::kMemcpyCPU2GPU,
                context.stream,
                true);
            op::RmsNormLayer gpu_layer(gpu_weight, epsilon);
            auto gpu_status = gpu_layer.forward({&gpu_x}, {&gpu_y}, context);
            ASSERT_EQ(gpu_status.get_err_code(), base::kSuccess) << gpu_status.get_err_message();

            cuda_allocator->memcpy(
                gpu_y.ptr<float>(),
                host_y.data(),
                gpu_y.byte_size(),
                base::MemcpyKind::kMemcpyGPU2CPU,
                context.stream,
                true);

            const auto expected = reference_rmsnorm(host_x, host_w, epsilon);
            expect_close(cpu_y.ptr<float>(), expected);
            expect_close(host_y.data(), expected);
            expect_close(host_y.data(), std::vector<float>(cpu_y.ptr<float>(), cpu_y.ptr<float>() + cpu_y.size()));

            // CPU 权重配 GPU 输入应在 Layer 层被拒绝。
            auto mixed = cpu_layer.forward({&gpu_x}, {&gpu_y}, context);
            EXPECT_EQ(mixed.get_err_code(), base::kInvalidArgument);
        }
    }
}