#include "model/qwen2.h"
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
const fs::path root =  QWEN_LAYER0_DATA_DIR;

tensor::Tensor read_f32(const std::string& name,
                        const std::vector<int64_t>& shape) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(std::endian::native == std::endian::little);
    if (shape.empty()) throw std::invalid_argument("Empty shape");
    std::size_t count = 1;
    for (auto d : shape) {
        if (d <= 0 || static_cast<std::uint64_t>(d) >
                          std::numeric_limits<std::size_t>::max() / count)
            throw std::invalid_argument("Invalid/overflowed shape: " + name);
        count *= static_cast<std::size_t>(d);
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        throw std::overflow_error("Byte size overflow");
    const auto bytes = count * sizeof(float);
    if (bytes > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max()))
        throw std::overflow_error("File too large");
    const auto path = root / (name + ".bin");
    if (fs::file_size(path) != bytes)
        throw std::runtime_error("File size mismatch: " + path.string());
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + path.string());
    tensor::Tensor result(shape, base::DataType::kDataTypeFp32, std::make_shared<base::CPUDeviceAllocator>());
    if (!file.read(reinterpret_cast<char*>(result.ptr<float>()),
                   static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("Short read: " + path.string());
    for (std::size_t i = 0; i < result.size(); ++i)
        if (!std::isfinite(result.ptr<float>()[i]))
            throw std::runtime_error("Non-finite data: " + name);
    return result;
}

std::shared_ptr<const tensor::Tensor> weight(
    const std::string& name,
    const std::vector<int64_t>& shape) {
    return std::make_shared<tensor::Tensor>(read_f32("w." + name, shape));
}

void expect_close(const tensor::Tensor& actual, const tensor::Tensor& expected, double atol = 1e-5, double rtol = 1e-4) {
    ASSERT_EQ(actual.dims(), expected.dims());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double a = actual.ptr<float>()[i], b = expected.ptr<float>()[i];
        ASSERT_TRUE(std::isfinite(a)) << "index=" << i;
        ASSERT_TRUE(std::isfinite(b)) << "index=" << i;
        ASSERT_LE(std::abs(a - b), atol + rtol * std::abs(b))
            << "index=" << i << " actual=" << a << " expected=" << b;
    }
}

model::Qwen2Config fixture_config() {
    model::Qwen2Config c;
    c.dim = 896;
    c.hidden_dim = 4864;
    c.q_heads = 14;
    c.kv_heads = 2;
    c.head_dim = 64;
    c.capacity = 8;
    c.epsilon = 1e-6F;
    c.rope_theta = 1000000.0;
    return c;
}

model::Qwen2AttentionWeights load_attention_weights(const model::Qwen2Config& c) {
    model::Qwen2AttentionWeights w;
    w.wq = weight("wq", {c.q_dim(), c.dim});
    w.wk = weight("wk", {c.kv_dim(), c.dim});
    w.wv = weight("wv", {c.kv_dim(), c.dim});
    w.wo = weight("wo", {c.dim, c.q_dim()});
    w.bq = weight("bq", {c.q_dim()});
    w.bk = weight("bk", {c.kv_dim()});
    w.bv = weight("bv", {c.kv_dim()});
    return w;
}

model::Qwen2LayerWeights load_layer_weights(const model::Qwen2Config& c) {
    model::Qwen2LayerWeights w;
    w.attention = load_attention_weights(c);
    w.attention_norm = weight("attention_norm", {c.dim});
    w.ffn_norm = weight("ffn_norm", {c.dim});
    w.gate = weight("gate", {c.hidden_dim, c.dim});
    w.up = weight("up", {c.hidden_dim, c.dim});
    w.down = weight("down", {c.dim, c.hidden_dim});
    return w;
}

} // namespace

TEST(Qwen2Layer0Test, AttentionRealWeightsCachedAndReset) {
    ASSERT_TRUE(fs::exists(root / "manifest.json"));
    const auto c = fixture_config();
    const auto w = load_attention_weights(c);
    model::Qwen2Attention attention(c, w);
    auto n = read_f32("ref.n", {5, c.dim});
    auto original_n = read_f32("ref.n", {5, c.dim});
    auto expected = read_f32("ref.attn_out", {5, c.dim});
    tensor::Tensor out({1, c.dim}, base::DataType::kDataTypeFp32, std::make_shared<base::CPUDeviceAllocator>());
    for (int request = 0; request < 2; ++request) {
        SCOPED_TRACE(request);
        attention.reset();
        ASSERT_EQ(attention.length(), 0);
        for (int64_t t = 0; t < 5; ++t) {
            SCOPED_TRACE(t);
            const auto offset = static_cast<std::size_t>(t * c.dim);
            auto nt = n.view({1, c.dim}, offset);
            auto et = expected.view({1, c.dim}, offset);
            const auto status = attention.forward(nt, out);
            ASSERT_TRUE(status) << status.get_err_message();
            ASSERT_NO_FATAL_FAILURE(expect_close(out, et));
            ASSERT_EQ(attention.length(), t + 1);
            ASSERT_FALSE(attention.failed());
        }
    }
    ASSERT_NO_FATAL_FAILURE(expect_close(n, original_n, 0, 0));
}
TEST(Qwen2Layer0Test, DecoderRealWeightsCachedAndReset) {
    ASSERT_TRUE(fs::exists(root / "manifest.json"));
    const auto c = fixture_config();
    const auto w = load_layer_weights(c);
    model::Qwen2DecoderLayer layer(c, w);
    auto x = read_f32("ref.x", {5, c.dim});
    auto original_x = read_f32("ref.x", {5, c.dim});
    auto expected = read_f32("ref.y", {5, c.dim});
    auto expected_cached = read_f32("ref.y_cached", {5, c.dim});
    tensor::Tensor y({1, c.dim}, base::DataType::kDataTypeFp32, std::make_shared<base::CPUDeviceAllocator>());
    for (int request = 0; request < 2; ++request) {
        SCOPED_TRACE(request);
        layer.reset();
        ASSERT_EQ(layer.length(), 0);
        for (int64_t t = 0; t < 5; ++t) {
            SCOPED_TRACE(t);
            const auto offset = static_cast<std::size_t>(t * c.dim);
            auto xt = x.view({1, c.dim}, offset);
            auto yt = expected.view({1, c.dim}, offset);
            auto yc = expected_cached.view({1, c.dim}, offset);
            const auto status = layer.forward(xt, y);
            ASSERT_TRUE(status) << status.get_err_message();
            ASSERT_NO_FATAL_FAILURE(expect_close(y, yt));
            ASSERT_NO_FATAL_FAILURE(expect_close(y, yc));
            ASSERT_EQ(layer.length(), t + 1);
            ASSERT_FALSE(layer.failed());
        }
    }
    ASSERT_NO_FATAL_FAILURE(expect_close(x, original_x, 0, 0));
}