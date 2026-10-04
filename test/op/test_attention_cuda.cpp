#include "base/kv_cache.h"
#include "op/attention.h"
#include "test_utils.h"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

// Attention 本课专用的固定输入生成规则，保留在本测试文件。
tensor::Tensor pattern(const std::vector<std::int64_t>& shape, int shift) {
    auto cpu = std::make_shared<base::CPUDeviceAllocator>();
    tensor::Tensor t(shape, base::DataType::kDataTypeFp32, cpu);
    for (std::size_t i = 0; i < t.size(); ++i) {
        t.ptr<float>()[i] = (static_cast<int>(i % 17) - 8 + shift) * 0.05F;
    }
    return t;
}

} // namespace


TEST(AttentionCudaTest, GqaKnownValues) {

    auto q = test_utils::make_cpu({4, 2}, {1, 0, 1, 0, 1, 0, 1, 0});
    auto k = test_utils::make_cpu({2, 2, 2}, {0, 0, 0, 0, 1, 0, -1, 0});
    auto v = test_utils::make_cpu({2, 2, 2}, {2, 4, 10, 20, 6, 8, 30, 40});


    auto q_gpu = test_utils::to_gpu(q);
    auto k_gpu = test_utils::to_gpu(k);
    auto v_gpu = test_utils::to_gpu(v);

    auto gpu = std::make_shared<base::CUDADeviceAllocator>();
    tensor::Tensor y_gpu({4, 2}, base::DataType::kDataTypeFp32, gpu);

    op::AttentionLayer attention;
    const auto status = attention.forward({&q_gpu, &k_gpu, &v_gpu}, {&y_gpu});
    ASSERT_TRUE(status) << status.get_err_message();

    const float p = static_cast<float>(1.0 / (1.0 + std::exp(-1.0 / std::sqrt(2.0))));
    const std::vector<float> expected{
        2 + 4*p, 4 + 4*p, 2 + 4*p, 4 + 4*p,
        10 + 20*(1-p), 20 + 20*(1-p), 10 + 20*(1-p), 20 + 20*(1-p)};
    test_utils::expect_near(y_gpu, expected, 1e-5, 0.0);

    // CPU Q 和 GPU K/V 混用，必须在进入 kernel 前被拒绝。
    EXPECT_EQ(attention.forward({&q, &k_gpu, &v_gpu}, {&y_gpu}).get_err_code(),
              base::kInvalidArgument);
}

TEST(AttentionCudaTest, QwenShapeWithIncrementalCache) {

    constexpr std::int64_t Hq = 14, Hkv = 2, D = 64, T = 3;

    auto gpu = std::make_shared<base::CUDADeviceAllocator>();
    auto cpu = std::make_shared<base::CPUDeviceAllocator>();

    base::KVCache cache(T + 1, Hkv, D, gpu); // 容量大于有效长度。
    auto all_k = pattern({T, Hkv, D}, 1);
    auto all_v = pattern({T, Hkv, D}, 3);

    tensor::Tensor y_cpu({Hq, D}, base::DataType::kDataTypeFp32, cpu);
    tensor::Tensor y_gpu({Hq, D}, base::DataType::kDataTypeFp32, gpu);

    op::AttentionLayer attention;

    for (std::int64_t position = 0; position < T; ++position) {
        SCOPED_TRACE(position);

        auto q_cpu = pattern({Hq, D}, static_cast<int>(position));
        auto q_gpu = test_utils::to_gpu(q_cpu);
        const auto offset = static_cast<std::size_t>(position * Hkv * D);
        auto k_now = all_k.view({Hkv, D}, offset);
        auto v_now = all_v.view({Hkv, D}, offset);
        auto k_slot = cache.key_slot(position);
        auto v_slot = cache.value_slot(position);
        test_utils::copy_fp32(k_now, k_slot);
        test_utils::copy_fp32(v_now, v_slot);
        ASSERT_TRUE(cache.commit(position));
        ASSERT_EQ(cache.length(), position + 1);

        auto k_cpu = all_k.view({position + 1, Hkv, D});
        auto v_cpu = all_v.view({position + 1, Hkv, D});
        auto k_gpu = cache.keys();
        auto v_gpu = cache.values();
        auto status = attention.forward({&q_cpu, &k_cpu, &v_cpu}, {&y_cpu});
        ASSERT_TRUE(status) << status.get_err_message();
        status = attention.forward({&q_gpu, &k_gpu, &v_gpu}, {&y_gpu});
        ASSERT_TRUE(status) << status.get_err_message();

        test_utils::expect_near(y_gpu, y_cpu, 1e-5, 1e-4);
    }
}