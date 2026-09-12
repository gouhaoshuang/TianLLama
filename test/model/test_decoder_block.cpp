#include "model/decoder_block.h"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using Row = std::vector<double>;
using Sequence = std::vector<Row>;

tensor::Tensor make_tensor(
    const std::vector<std::int64_t>& shape,
    const Row& data) {
    tensor::Tensor t(shape, base::DataType::kDataTypeFp32, std::make_shared<base::CPUDeviceAllocator>());
    if (t.size() != data.size()) throw std::invalid_argument("Test data size");
    std::copy(data.begin(), data.end(), t.ptr<float>());
    return t;
}

std::shared_ptr<const tensor::Tensor> constant_weight(
    const std::vector<std::int64_t>& shape,
    double value) {

    std::size_t count = 1;
    for (auto d : shape)
        count *= static_cast<std::size_t>(d);
    return std::make_shared<tensor::Tensor>(make_tensor(shape, Row(count, value)));
}

model::DecoderWeights fixed_weights(
    const model::DecoderConfig& c,
    int seed = 1) {

    auto matrix = [&](int rows, int cols, int salt) {
        Row data(rows * cols);
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = ((i * 7 + seed * 3 + salt * 5) % 19 - 9) * 0.035;
        }
        return std::make_shared<tensor::Tensor>(make_tensor({rows, cols}, data));
    };
    Row norm1(c.dim), norm2(c.dim);
    for (int i = 0; i < c.dim; ++i) {
        norm1[i] = 0.9 + 0.04 * (i % 3);
        norm2[i] = 1.1 - 0.03 * (i % 4);
    }
    model::DecoderWeights w;
    w.attention_norm = std::make_shared<tensor::Tensor>(make_tensor({c.dim}, norm1));
    w.ffn_norm = std::make_shared<tensor::Tensor>(make_tensor({c.dim}, norm2));
    w.wq = matrix(c.dim, c.dim, 1);
    w.wk = matrix(c.dim, c.dim, 2);
    w.wv = matrix(c.dim, c.dim, 3);
    w.wo = matrix(c.dim, c.dim, 4);
    w.gate = matrix(c.hidden_dim, c.dim, 5);
    w.up = matrix(c.hidden_dim, c.dim, 6);
    w.down = matrix(c.dim, c.hidden_dim, 7);
    return w;
}

Sequence sample_input() {
    return {{1, -0.5, 0.25, 2, -1, 0.5, 1.5, -0.25},
            {-0.25, 1, 0.5, -1, 2, -0.5, 0.25, 1.5},
            {0.5, 0.25, -1, 1.5, -0.25, 2, -0.5, 1}};
}

void expect_row(const tensor::Tensor& actual, const Row& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual.ptr<float>()[i], expected[i], 2e-5 + 2e-5 * std::abs(expected[i])) << "channel=" << i;
    }
}
} // namespace

TEST(DecoderBlockTest, ZeroBranchesPreserveResidual) {
    model::DecoderConfig c;
    auto w = fixed_weights(c);

    w.wo = constant_weight({c.dim, c.dim}, 0);
    w.down = constant_weight({c.dim, c.hidden_dim}, 0);

    model::DecoderBlock block(c, w);

    const auto inputs = sample_input();

    auto y = make_tensor({1, c.dim}, Row(c.dim, 123));

    for (std::size_t t = 0; t < inputs.size(); ++t) {

        auto x = make_tensor({1, c.dim}, inputs[t]);

        const auto status = block.forward(x, y);
        ASSERT_TRUE(status) << status.get_err_message();

        expect_row(y, inputs[t]);
        expect_row(x, inputs[t]);

        EXPECT_EQ(block.length(), static_cast<std::int64_t>(t + 1));
    }
}


namespace {
Row ref_norm(const Row& x, const tensor::Tensor& gamma, double eps) {
    double square_sum = 0;
    for (double v : x) square_sum += v * v;
    const double scale = 1 / std::sqrt(square_sum / x.size() + eps);
    Row y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] = x[i] * scale * gamma.ptr<float>()[i];
    }
    return y;
}

Row ref_linear(const Row& x, const tensor::Tensor& w) {
    Row y(w.dim(0), 0);
    for (int n = 0; n < w.dim(0); ++n) {
        for (int k = 0; k < w.dim(1); ++k) {
            y[n] += x[k] * w.ptr<float>()[n * w.dim(1) + k];
        }
    }
    return y;
}

void ref_rope(Row& x, int position, int H, int D, double theta) {
    for (int h = 0; h < H; ++h) {
        for (int d = 0; d < D; d += 2) {
            const double angle = position * std::pow(theta, -double(d) / D);
            const int i = h * D + d;
            const double a = x[i], b = x[i + 1];
            x[i] = a * std::cos(angle) - b * std::sin(angle);
            x[i + 1] = a * std::sin(angle) + b * std::cos(angle);
        }
    }
}

Sequence full_block_reference(const Sequence& x, const model::DecoderConfig& c,
                              const model::DecoderWeights& w) {
    const int T = static_cast<int>(x.size());
    const int D = static_cast<int>(c.dim / c.heads);
    Sequence q(T), k(T), v(T), output(T);
    for (int t = 0; t < T; ++t) {
        const auto n = ref_norm(x[t], *w.attention_norm, c.epsilon);
        q[t] = ref_linear(n, *w.wq);
        k[t] = ref_linear(n, *w.wk);
        v[t] = ref_linear(n, *w.wv);
        ref_rope(q[t], t, c.heads, D, c.rope_theta);
        ref_rope(k[t], t, c.heads, D, c.rope_theta);
    }
    for (int t = 0; t < T; ++t) {
        Row a(c.dim, 0);
        for (int h = 0; h < c.heads; ++h) {
            Row scores(T, -std::numeric_limits<double>::infinity());
            for (int s = 0; s <= t; ++s) {
                double dot = 0;
                for (int d = 0; d < D; ++d) dot += q[t][h * D + d] * k[s][h * D + d];
                scores[s] = dot / std::sqrt(double(D));
            }
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double denominator = 0;
            for (double& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (int s = 0; s < T; ++s) {
                for (int d = 0; d < D; ++d) {
                    a[h * D + d] += scores[s] / denominator * v[s][h * D + d];
                }
            }
        }
        Row h = ref_linear(a, *w.wo);
        for (int i = 0; i < c.dim; ++i) h[i] += x[t][i];
        const auto z = ref_norm(h, *w.ffn_norm, c.epsilon);
        auto gate = ref_linear(z, *w.gate);
        const auto up = ref_linear(z, *w.up);
        for (int i = 0; i < c.hidden_dim; ++i) {
            const double e = std::exp(-std::abs(gate[i]));
            const double sigmoid = gate[i] >= 0 ? 1 / (1 + e) : e / (1 + e);
            gate[i] = gate[i] * sigmoid * up[i];
        }
        output[t] = ref_linear(gate, *w.down);
        for (int i = 0; i < c.dim; ++i) output[t][i] += h[i];
    }
    return output;
}
} // namespace

TEST(DecoderBlockTest, IncrementalMatchesFullCausalAndReset) {
    model::DecoderConfig c;
    c.rope_theta = 128; // 故意不用默认值，检查参数有没有真正传入 RoPE。
    auto w = fixed_weights(c);
    model::DecoderBlock block(c, w);
    const auto inputs = sample_input();
    const auto expected = full_block_reference(inputs, c, w);
    auto y = make_tensor({1, c.dim}, Row(c.dim, 123));

    for (int request = 0; request < 2; ++request) {
        SCOPED_TRACE(request);
        block.reset();
        ASSERT_EQ(block.length(), 0);
        for (std::size_t t = 0; t < inputs.size(); ++t) {
            SCOPED_TRACE(t);
            auto x = make_tensor({1, c.dim}, inputs[t]);
            const auto status = block.forward(x, y);
            ASSERT_TRUE(status) << status.get_err_message();
            expect_row(y, expected[t]);
            expect_row(x, inputs[t]);
            EXPECT_EQ(block.length(), static_cast<std::int64_t>(t + 1));
            EXPECT_FALSE(block.failed());
        }
    }
}

TEST(DecoderBlockTest, TwoBlocksHaveIndependentCaches) {
    model::DecoderConfig c;
    auto w1 = fixed_weights(c, 1);
    auto w2 = fixed_weights(c, 2); // 两层权重不同。
    model::DecoderBlock first(c, w1), second(c, w2);
    const auto inputs = sample_input();
    const auto expected1 = full_block_reference(inputs, c, w1);
    const auto expected2 = full_block_reference(expected1, c, w2);
    auto middle = make_tensor({1, c.dim}, Row(c.dim, 0));
    auto y = make_tensor({1, c.dim}, Row(c.dim, 0));

    for (std::size_t t = 0; t < inputs.size(); ++t) {
        SCOPED_TRACE(t);
        auto x = make_tensor({1, c.dim}, inputs[t]);
        auto status = first.forward(x, middle);
        ASSERT_TRUE(status) << status.get_err_message();
        status = second.forward(middle, y);
        ASSERT_TRUE(status) << status.get_err_message();
        expect_row(middle, expected1[t]);
        expect_row(y, expected2[t]);
        EXPECT_EQ(first.length(), static_cast<std::int64_t>(t + 1));
        EXPECT_EQ(second.length(), static_cast<std::int64_t>(t + 1));
    }
}