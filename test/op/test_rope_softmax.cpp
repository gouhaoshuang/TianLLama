#include "op/rope.h"
#include "op/softmax.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

tensor::Tensor copy_to_gpu(const tensor::Tensor& cpu) {
    auto allocator = std::make_shared<base::CUDADeviceAllocator>();
    tensor::Tensor gpu(cpu.dims(), cpu.data_type(), allocator);
    allocator->memcpy(cpu.ptr<float>(), gpu.ptr<float>(), cpu.byte_size(), base::MemcpyKind::kMemcpyCPU2GPU, nullptr, true);
    return gpu;
}

tensor::Tensor copy_to_cpu(const tensor::Tensor& gpu) {
    auto allocator = std::make_shared<base::CPUDeviceAllocator>();
    tensor::Tensor cpu(gpu.dims(), gpu.data_type(), allocator);
    allocator->memcpy(gpu.ptr<float>(), cpu.ptr<float>(), gpu.byte_size(), base::MemcpyKind::kMemcpyGPU2CPU, nullptr, true);
    return cpu;
}

tensor::Tensor make_input(
    const std::vector<std::int64_t>& shape,
    const std::vector<float>& values) {

    auto cpu = std::make_shared<base::CPUDeviceAllocator>();

    tensor::Tensor t(shape, base::DataType::kDataTypeFp32, cpu);

    if (t.size() != values.size()) {
        throw std::invalid_argument("Test data size mismatch");
    }
    std::copy(values.begin(), values.end(), t.ptr<float>());
    return t; // Tensor 支持移动，不要求拷贝。
}

void expect_values(const tensor::Tensor& t, const std::vector<float>& expected) {

    ASSERT_EQ(t.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(t.ptr<float>()[i], expected[i], 1e-5F) << "index=" << i;
    }
}

} // namespace

TEST(ROPETest, PositionZero) {
    auto x = make_input({2, 4}, {1, 2, 3, 4, -1, 0.5F, 2, -3});
    auto y = make_input({2, 4}, {0, 0, 0, 0, 0, 0, 0, 0});
    op::RoPELayer rope(0);
    const base::Status status = rope.forward({&x}, {&y});

    ASSERT_TRUE(status) << status.get_err_message();

    expect_values(y, {1, 2, 3, 4, -1, 0.5F, 2, -3});
}
TEST(RoPETest, PositionOneAndPairNorm) {

    auto x = make_input({2, 4}, {1, 2, 3, 4, -1, 0.5F, 2, -3});
    auto y = make_input({2, 4}, {0, 0, 0, 0, 0, 0, 0, 0});

    op::RoPELayer rope(1, 10000.0);

    const auto status = rope.forward({&x}, {&y});

    ASSERT_TRUE(status) << status.get_err_message();

    // 两个 head 都使用角度 1 和 0.01，参考答案直接由二维旋转公式计算。
    expect_values(y, {-1.142639664F, 1.922075597F, 2.959850668F, 4.029799502F, -0.961037798F, -0.571319832F, 2.029899501F, -2.979850335F});

    expect_values(x, {1, 2, 3, 4, -1, 0.5F, 2, -3});

    for (std::size_t i = 0; i < x.size(); i += 2) {

        const double a = x.ptr<float>()[i], b = x.ptr<float>()[i + 1];
        const double c = y.ptr<float>()[i], d = y.ptr<float>()[i + 1];

        EXPECT_NEAR(a * a + b * b, c * c + d * d, 1e-5) << "pair=" << i / 2;
    }
}

TEST(SoftmaxTest, StableRows) {
    // 三行只相差一个常数，Softmax 应相同；不能把整个 Tensor 一起归一化。
    auto x = make_input({3, 3}, {1, 2, 3, 1001, 1002, 1003, -1001, -1000, -999});
    auto y = make_input({3, 3}, {0, 0, 0, 0, 0, 0, 0, 0, 0});

    op::SoftmaxLayer softmax;
    const auto status = softmax.forward({&x}, {&y});

    ASSERT_TRUE(status) << status.get_err_message();
    expect_values(y, {0.090030573F, 0.244728471F, 0.665240956F, 0.090030573F, 0.244728471F, 0.665240956F, 0.090030573F, 0.244728471F, 0.665240956F});

    for (std::size_t r = 0; r < 3; ++r) {
        float sum = 0.0F;
        for (std::size_t j = 0; j < 3; ++j) {
            const float p = y.ptr<float>()[r * 3 + j];
            EXPECT_TRUE(std::isfinite(p));
            EXPECT_GE(p, 0.0F);
            EXPECT_LE(p, 1.0F);
            sum += p;
        }
        EXPECT_NEAR(sum, 1.0F, 1e-6F);
    }

    expect_values(x, {1, 2, 3, 1001, 1002, 1003, -1001, -1000, -999});
}

TEST(SoftmaxTest, UniformAndSingleColumn) {
    op::SoftmaxLayer softmax;
    auto x = make_input({1, 3}, {7, 7, 7});
    auto y = make_input({1, 3}, {0, 0, 0});
    ASSERT_TRUE(softmax.forward({&x}, {&y}));
    expect_values(y, {1.0F / 3, 1.0F / 3, 1.0F / 3});

    auto single = make_input({2, 1}, {1000, -1000});
    auto single_y = make_input({2, 1}, {0, 0});
    ASSERT_TRUE(softmax.forward({&single}, {&single_y}));
    expect_values(single_y, {1, 1});
}

TEST(SoftmaxTest, RejectsInvalidInput) {
    op::SoftmaxLayer softmax;
    auto x = make_input({1, 3}, {1, 2, 3});
    auto y = make_input({1, 3}, {123, 123, 123});
    auto wrong = make_input({3, 1}, {0, 0, 0});
    EXPECT_EQ(softmax.forward({&x}, {&wrong}).get_err_code(), base::kInvalidArgument);
    EXPECT_EQ(softmax.forward({&x}, {&x}).get_err_code(), base::kInvalidArgument);
    EXPECT_EQ(softmax.forward({nullptr}, {&y}).get_err_code(), base::kInvalidArgument);

    const float inf = std::numeric_limits<float>::infinity();
    for (float bad : {inf, -inf, std::numeric_limits<float>::quiet_NaN()}) {
        x.ptr<float>()[2] = bad;
        EXPECT_EQ(softmax.forward({&x}, {&y}).get_err_code(), base::kInvalidArgument);
        expect_values(y, {123, 123, 123}); // 检查失败时尚未写输出。
    }
}

TEST(CudaRoPESoftmaxTest, RoPEFixedValues) {
    auto cpu_x = make_input({2, 4}, {1, 2, 3, 4, -1, 0.5F, 2, -3});
    auto x = copy_to_gpu(cpu_x);
    auto y = copy_to_gpu(make_input({2, 4}, {0, 0, 0, 0, 0, 0, 0, 0}));

    op::RoPELayer rope(1);
    const auto status = rope.forward({&x}, {&y});
    ASSERT_TRUE(status) << status.get_err_message();

    auto result = copy_to_cpu(y);
    expect_values(result, {-1.142639664F, 1.922075597F, 2.959850668F, 4.029799502F,
                           -0.961037798F, -0.571319832F, 2.029899501F, -2.979850335F});

    auto input_after = copy_to_cpu(x);
    expect_values(input_after, {1, 2, 3, 4, -1, 0.5F, 2, -3});
}

TEST(CudaRoPESoftmaxTest, SoftmaxStableRows) {
    auto cpu_x = make_input({2, 3}, {1001, 1002, 1003, -1001, -1000, -999});
    auto x = copy_to_gpu(cpu_x);
    auto y = copy_to_gpu(make_input({2, 3}, {0, 0, 0, 0, 0, 0}));

    op::SoftmaxLayer softmax;
    const auto status = softmax.forward({&x}, {&y});
    ASSERT_TRUE(status) << status.get_err_message();

    auto result = copy_to_cpu(y);
    expect_values(result, {0.090030573F, 0.244728471F, 0.665240956F,
                           0.090030573F, 0.244728471F, 0.665240956F});
    for (int row = 0; row < 2; ++row) {
        const float* p = result.ptr<float>() + row * 3;
        EXPECT_NEAR(p[0] + p[1] + p[2], 1.0F, 1e-5F);
    }

    auto input_after = copy_to_cpu(x);
    expect_values(input_after, {1001, 1002, 1003, -1001, -1000, -999});
}

TEST(CudaRoPESoftmaxTest, RejectsNaN) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto x = copy_to_gpu(make_input({1, 4}, {1, 2, nan, 4}));
    auto y = copy_to_gpu(make_input({1, 4}, {123, 123, 123, 123}));

    op::RoPELayer rope(1);
    EXPECT_EQ(rope.forward({&x}, {&y}).get_err_code(), base::kInvalidArgument);
    auto after_rope = copy_to_cpu(y);
    expect_values(after_rope, {123, 123, 123, 123});

    op::SoftmaxLayer softmax;
    EXPECT_EQ(softmax.forward({&x}, {&y}).get_err_code(), base::kInvalidArgument);
    auto after_softmax = copy_to_cpu(y);
    expect_values(after_softmax, {123, 123, 123, 123});
}