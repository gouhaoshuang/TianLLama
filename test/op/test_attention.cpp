#include "base/kv_cache.h"
#include "op/attention.h"

#include "op/rope.h"
#include "op/softmax.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

tensor::Tensor make_data(const std::vector<std::int64_t>& shape,
                         const std::vector<float>& values) {

    tensor::Tensor t(shape, base::DataType::kDataTypeFp32, std::make_shared<base::CPUDeviceAllocator>());

    if (t.size() != values.size())
        throw std::invalid_argument("Test shape mismatch");
    std::copy(values.begin(), values.end(), t.ptr<float>());
    return t;
}

void expect_data(const tensor::Tensor& t, const std::vector<float>& values) {
    ASSERT_EQ(t.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        EXPECT_NEAR(t.ptr<float>()[i], values[i], 1e-5F) << "index=" << i;
    }
}

using Tokens = std::vector<std::vector<float>>; // 每行是一个 token 的 H*D 个元素。

Tokens full_causal_reference(Tokens q, Tokens k, const Tokens& v,
                             int H, int D) {
    const int T = static_cast<int>(q.size());
    auto rotate_all = [=](Tokens& data) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < H; ++h) {
                for (int d = 0; d < D; d += 2) {
                    const double angle = t * std::pow(10000.0, -double(d) / D);
                    const double c = std::cos(angle), s = std::sin(angle);
                    const int i = h * D + d;
                    const double a = data[t][i], b = data[t][i + 1];
                    data[t][i] = static_cast<float>(a * c - b * s);
                    data[t][i + 1] = static_cast<float>(a * s + b * c);
                }
            }
        }
    };
    rotate_all(q);
    rotate_all(k);
    Tokens output(T, std::vector<float>(H * D, 0));
    for (int h = 0; h < H; ++h) {
        std::vector<double> scores(T * T);
        for (int t = 0; t < T; ++t) {
            for (int s = 0; s < T; ++s) {
                double dot = 0;
                for (int d = 0; d < D; ++d) {
                    dot += double(q[t][h * D + d]) * k[s][h * D + d];
                }
                scores[t * T + s] = s > t
                    ? -std::numeric_limits<double>::infinity()
                    : dot / std::sqrt(double(D));
            }
        }
        for (int t = 0; t < T; ++t) {
            auto first = scores.begin() + t * T;
            const double maximum = *std::max_element(first, first + T);
            double sum = 0;
            for (int s = 0; s < T; ++s) {
                scores[t * T + s] = std::exp(scores[t * T + s] - maximum);
                sum += scores[t * T + s];
            }
            for (int d = 0; d < D; ++d) {
                double value = 0;
                for (int s = 0; s < T; ++s) {
                    value += scores[t * T + s] / sum * v[s][h * D + d];
                }
                output[t][h * D + d] = static_cast<float>(value);
            }
        }
    }
    return output;
}

} // namespace

TEST(AttentionTest, TensorViewSafety) {
    tensor::Tensor parent = make_data({8}, {0, 1, 2, 3, 4, 5, 6, 7});
    tensor::Tensor a = parent.view({1, 4}, 0);
    tensor::Tensor b = parent.view({1, 4}, 1);

    tensor::Tensor separate = parent.view({1, 4}, 4);

    EXPECT_EQ(a.byte_size(), 4 * sizeof(float));
    EXPECT_EQ(b.ptr<float>(), parent.ptr<float>() + 1);

    EXPECT_TRUE(a.overlaps(b));

    EXPECT_FALSE(a.overlaps(separate));

    b.ptr<float>()[0] = 20;
    EXPECT_FLOAT_EQ(parent.ptr<float>()[1], 20);
    EXPECT_THROW(parent.view({4}, 5), std::out_of_range);
    EXPECT_THROW(a.view({4}, 1), std::out_of_range); // 不得逃出父视图。

    op::SoftmaxLayer softmax;
    EXPECT_EQ(softmax.forward({&a}, {&b}).get_err_code(), base::kInvalidArgument);

    // 父 Tensor 已销毁，视图仍持有 Buffer，不能悬空。
    auto survivor = [] {
        auto local = make_data({4}, {1, 2, 3, 4});
        return local.view({1, 2}, 2);
    }();
    expect_data(survivor, {3, 4});
}

TEST(AttentionTest, CacheLifecycle) {
    // 1. cache 创建
    base::KVCache cache(2, 1, 2);

    EXPECT_EQ(cache.length(), 0);

    EXPECT_THROW(cache.keys(), std::logic_error);
    EXPECT_THROW(cache.key_slot(1), std::out_of_range);

    // 2. cache 写入
    tensor::Tensor k0 = cache.key_slot(0);
    tensor::Tensor v0 = cache.value_slot(0);

    k0.ptr<float>()[0] = 1;
    k0.ptr<float>()[1] = 2;
    v0.ptr<float>()[0] = 3;
    v0.ptr<float>()[1] = 4;

    EXPECT_EQ(cache.length(), 0); // 写完之后不会自动提交
    EXPECT_TRUE(cache.commit(0));
    EXPECT_EQ(cache.length(), 1);
    EXPECT_THROW(cache.key_slot(0), std::out_of_range);

    // 3. 写入NaN
    tensor::Tensor k1 = cache.key_slot(1);
    tensor::Tensor v1 = cache.value_slot(1);
    k1.ptr<float>()[0] = 5;
    k1.ptr<float>()[1] = 6;
    v1.ptr<float>()[0] = std::numeric_limits<float>::quiet_NaN();
    v1.ptr<float>()[1] = 8;

    EXPECT_EQ(cache.commit(1).get_err_code(), base::kInvalidArgument);
    EXPECT_EQ(cache.length(), 1);
    v1.ptr<float>()[0] = 7;
    ASSERT_TRUE(cache.commit(1));
    expect_data(cache.keys(), {1, 2, 5, 6});
    expect_data(cache.values(), {3, 4, 7, 8});

    EXPECT_THROW(cache.key_slot(2), std::out_of_range); // cache 容量为2 ， 此时cache 已经存满
    EXPECT_EQ(cache.commit(2).get_err_code(), base::kInvalidArgument);
    EXPECT_EQ(cache.length(), 2);

    const float* storage_start = k0.ptr<float>();
    cache.reset();
    EXPECT_EQ(cache.length(), 0);
    tensor::Tensor new_k = cache.key_slot(0);
    tensor::Tensor new_v = cache.value_slot(0);

    EXPECT_EQ(new_k.ptr<float>(), storage_start);
    new_k.ptr<float>()[0] = 0;
    new_k.ptr<float>()[1] = 1;
    new_v.ptr<float>()[0] = 30;
    new_v.ptr<float>()[1] = 40;
    ASSERT_TRUE(cache.commit(0));

    auto q = make_data({1, 2}, {1, 0});
    auto y = make_data({1, 2}, {123, 123});
    auto keys = cache.keys();
    auto values = cache.values();
    op::AttentionLayer attention;
    ASSERT_TRUE(attention.forward({&q, &keys, &values}, {&y}));
    expect_data(y, {30, 40}); // 只有一个有效 token，不能混入旧请求的第二个槽位。
}


TEST(AttentionTest, IncrementalMatchesFullCausal) {
    const int H = 2, D = 4;
    const Tokens q = {
        {1, 0, 0, 1,   1, 1, 0, 1},
        {0, 1, 1, 0,   1,-1, 1, 0},
        {1, 1, 0,-1,   0, 1, 1, 1}
    };
    const Tokens k = {
        {1, 0, 1, 0,   0, 1, 0, 1},
        {0, 1, 0, 1,   1, 0, 1, 0},
        {1,-1, 1, 1,   1, 1,-1, 0}
    };
    const Tokens v = {
        {1, 2, 3, 4,   2, 0, 1,-1},
        {4, 3, 2, 1,  -1, 2, 0, 3},
        {2,-1, 0, 5,   3, 1,-2, 0}
    };
    const auto expected = full_causal_reference(q, k, v, H, D);
    base::KVCache cache(4, H, D); // 容量故意大于序列长度 3。
    op::AttentionLayer attention;
    auto q_rot = make_data({H, D}, std::vector<float>(H * D, 0));
    auto output = make_data({H, D}, std::vector<float>(H * D, 123));

    for (int position = 0; position < 3; ++position) {
        auto q_now = make_data({H, D}, q[position]);
        auto k_now = make_data({H, D}, k[position]);
        auto k_slot = cache.key_slot(position);
        auto v_slot = cache.value_slot(position);
        op::RoPELayer rope(position);
        ASSERT_TRUE(rope.forward({&q_now}, {&q_rot}));
        ASSERT_TRUE(rope.forward({&k_now}, {&k_slot}));
        
        std::copy_n(v[position].data(), H * D, v_slot.ptr<float>());
        ASSERT_TRUE(cache.commit(position));
        ASSERT_EQ(cache.length(), position + 1);

        auto keys = cache.keys();
        auto values = cache.values();
        const auto status = attention.forward({&q_rot, &keys, &values}, {&output});
        ASSERT_TRUE(status) << status.get_err_message();
        expect_data(output, expected[position]);
        if (position == 0) expect_data(output, v[0]); // 一个 key 的概率必为 1。
    }
}

TEST(AttentionTest, GqaSmallValues) {
    auto q = make_data({4,2}, {1,0, 1,0, 1,0, 1,0});
    // 按 token 排列：每个 token 内先 KV 头 0，再 KV 头 1。
    auto k = make_data({2,2,2}, {
        0,0,  0,0,   // token 0
        1,0, -1,0    // token 1
    });
    auto v = make_data({2,2,2}, {
        2,4, 10,20,  // token 0
        6,8, 30,40   // token 1
    });
    auto y = make_data({4,2}, {123,123,123,123,123,123,123,123});

    op::AttentionLayer attention;
    const auto status = attention.forward({&q, &k, &v}, {&y});
    ASSERT_TRUE(status) << status.get_err_message();

    const float p = static_cast<float>(1.0 / (1.0 + std::exp(-1.0 / std::sqrt(2.0))));
    const float a = 2 + 4*p, b = 4 + 4*p;
    const float c = 10 + 20*(1-p), d = 20 + 20*(1-p);
    expect_data(y, {a,b, a,b, c,d, c,d});
}
TEST(AttentionTest, MqaAveragesTwoTokens) {
    auto q = make_data({4,2}, {0,0, 0,0, 0,0, 0,0});
    auto k = make_data({2,1,2}, {1,2, 3,4});
    auto v = make_data({2,1,2}, {2,4, 6,8});
    auto y = make_data({4,2}, {123,123,123,123,123,123,123,123});

    op::AttentionLayer attention;
    const auto status = attention.forward({&q, &k, &v}, {&y});
    ASSERT_TRUE(status) << status.get_err_message();
    expect_data(y, {4,6, 4,6, 4,6, 4,6});
}