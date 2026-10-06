#include "sampler/greedy_sampler.h"
#include "test_utils.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <vector>

TEST(GreedySamplerTest, CpuOnly) {
    sampler::GreedySampler cpu(base::DeviceType::kDeviceCPU);
    auto logits = test_utils::make_cpu({1, 4}, {-9, -2, -2, -5});
    int32_t id = -1;
    ASSERT_TRUE(cpu.sample(logits, id));
    EXPECT_EQ(id, 1); // 全负数且最大值重复，取较小 ID。

    auto invalid = test_utils::make_cpu(
        {1, 2}, {1, std::numeric_limits<float>::quiet_NaN()});
    EXPECT_EQ(cpu.sample(invalid, id).get_err_code(), base::kInternalError);
    EXPECT_EQ(id, 1); // 失败不修改输出。

    ASSERT_TRUE(cpu.sample(logits, id));
    EXPECT_EQ(id, 1); // 失败后仍可正常采样。
}

TEST(GreedySamplerTest, CpuGpuAgreeAndReuse) {
    sampler::GreedySampler cpu(base::DeviceType::kDeviceCPU);
    sampler::GreedySampler gpu(base::DeviceType::kDeviceGPU);

    auto check = [&](const std::vector<float>& values, int32_t expected) {
        auto x = test_utils::make_cpu({1, static_cast<int64_t>(values.size())}, values);
        auto x_gpu = test_utils::to_gpu(x);
        int32_t a = -1, b = -1;
        auto status = cpu.sample(x, a);
        ASSERT_TRUE(status) << status.get_err_message();
        status = gpu.sample(x_gpu, b);
        ASSERT_TRUE(status) << status.get_err_message();
        EXPECT_EQ(a, expected);
        EXPECT_EQ(b, expected);
    };

    ASSERT_NO_FATAL_FAILURE(check({-3}, 0));
    ASSERT_NO_FATAL_FAILURE(check({-9, -2, -5}, 1));
    ASSERT_NO_FATAL_FAILURE(check({1, 7, 7, 2}, 1)); // 平局取小 ID。
    std::vector<float> tail(257, -4.0F);
    tail.back() = 3.0F;
    ASSERT_NO_FATAL_FAILURE(check(tail, 256)); // 非整 block，最大值在尾部。
    std::vector<float> vocab(151936, -1.0F);
    vocab.back() = 2.0F;
    ASSERT_NO_FATAL_FAILURE(check(vocab, 151935));
    ASSERT_NO_FATAL_FAILURE(check({2, 1}, 0)); // 同一个 GPU 结果 Buffer 再次复用。
}

TEST(GreedySamplerTest, RejectInvalidLogitsAndRecover) {
    sampler::GreedySampler cpu(base::DeviceType::kDeviceCPU);
    sampler::GreedySampler gpu(base::DeviceType::kDeviceGPU);
    for (float bad : {std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity()}) {
        auto x = test_utils::make_cpu({1, 3}, {5, bad, 1});
        auto x_gpu = test_utils::to_gpu(x);
        int32_t a = -7, b = -7;
        EXPECT_EQ(cpu.sample(x, a).get_err_code(), base::kInternalError);
        EXPECT_EQ(gpu.sample(x_gpu, b).get_err_code(), base::kInternalError);
        EXPECT_EQ(a, -7);
        EXPECT_EQ(b, -7); // 失败不写调用者输出。
    }

    auto valid = test_utils::make_cpu({1, 2}, {1, 2});
    auto valid_gpu = test_utils::to_gpu(valid);
    int32_t id = -1;
    ASSERT_TRUE(gpu.sample(valid_gpu, id)); // 上一次 invalid 标志不会残留。
    EXPECT_EQ(id, 1);
    EXPECT_EQ(gpu.sample(valid, id).get_err_code(), base::kInvalidArgument);
    auto wrong_shape = test_utils::make_cpu({2, 2}, {1, 2, 3, 4});
    EXPECT_EQ(cpu.sample(wrong_shape, id).get_err_code(), base::kInvalidArgument);
}