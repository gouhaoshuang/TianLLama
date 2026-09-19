
#include "base/utils.h"
#include "model/qwen2.h"
#include "tokenizer/qwen_tokenizer.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

TEST(QwenModelRealTest, ShortLogitsAndReset) {
    const std::filesystem::path root = QWEN_MODEL_DATA_DIR;
    std::ifstream file(root / "manifest.json");

    ASSERT_TRUE(file) << "打开 manifest.json 失败！" << root;

    const auto manifest = nlohmann::json::parse(file);

    ASSERT_EQ(manifest.at("format_version"), 2);

    const auto ids = manifest
                         .at("references")
                         .at("short")
                         .at("token_ids")
                         .get<std::vector<int32_t>>();

    const auto T = static_cast<int64_t>(ids.size());
    const auto V = manifest.at("config").at("vocab_size").get<int64_t>();
    ASSERT_GT(V, 0);

    const auto& entry = manifest.at("reference_tensors").at("ref.short.logits");
    ASSERT_EQ(entry.at("file"), "ref.short.logits.bin");
    ASSERT_EQ(entry.at("dtype"), "<f4");
    ASSERT_EQ(entry.at("shape"), nlohmann::json(std::vector<int64_t>{T, V}));

    auto expected = base::read_f32(root / "ref.short.logits.bin", {T, V});

    std::unique_ptr<model::Qwen2Model> model = model::Qwen2Model::load(root, T);

    // 一个局部函数：执行位置 t 的 token，并比较这一行完整的词表分数。
    auto check_token = [&](int64_t t) {
        const auto status = model->forward_token(ids[t]);
        ASSERT_TRUE(status) << status.get_err_message();
        ASSERT_FALSE(model->failed());
        ASSERT_EQ(model->length(), t + 1);
        for (int64_t i = 0; i < model->config().num_layers; ++i)
            ASSERT_EQ(model->layer_length(i), t + 1) << "layer=" << i;

        const auto& actual = model->logits();
        ASSERT_EQ(actual.dims(), (std::vector<int64_t>{1, V}));
        const float* a = actual.ptr<float>();
        const float* e = expected.ptr<float>() + t * V;
        for (int64_t v = 0; v < V; ++v) {
            ASSERT_TRUE(std::isfinite(a[v])) << "position=" << t << " vocab_id=" << v;
            // 初始阈值，尚不代表当前实现已经实测通过。
            const double tolerance = 1e-3 + 1e-4 * std::abs(double(e[v]));
            ASSERT_LE(std::abs(double(a[v]) - double(e[v])), tolerance)
                << "position=" << t << " vocab_id=" << v
                << " actual=" << a[v] << " expected=" << e[v];
        }
    };

    // 不在 token 之间 reset：后续位置必须能使用之前的 KV Cache。
    for (int64_t t = 0; t < T; ++t)
        ASSERT_NO_FATAL_FAILURE(check_token(t));

    // 同一个模型开启新请求，只重测首个 token，避免重复跑完整序列。
    model->reset();
    ASSERT_FALSE(model->failed());
    ASSERT_EQ(model->length(), 0);
    for (int64_t i = 0; i < model->config().num_layers; ++i)
        ASSERT_EQ(model->layer_length(i), 0);
    ASSERT_NO_FATAL_FAILURE(check_token(0));
}

TEST(QwenModelRealTest, ChatTokenizerAndLogits) {
    const std::filesystem::path root = QWEN_MODEL_DATA_DIR;

    // 1. 读取模型索引和官方聊天模板参考。
    std::ifstream manifest_file(root / "manifest.json");
    std::ifstream cases_file(root / "tokenizer_cases.json");
    ASSERT_TRUE(manifest_file);
    ASSERT_TRUE(cases_file);

    const auto manifest = nlohmann::json::parse(manifest_file);
    const auto cases = nlohmann::json::parse(cases_file);
    ASSERT_EQ(manifest.at("format_version"), 2);

    // 2. 真正通过 C++ 聊天模板和 Tokenizer 产生输入。
    tokenizer::QwenTokenizer tok((root / "tokenizer.json").string());

    const auto prompt = tokenizer::format_qwen_prompt("你好");
    ASSERT_EQ(prompt, cases.at("chat_text").get<std::string>());

    const auto ids = tok.encode(prompt);
    const auto reference_ids = manifest.at("references")
                                   .at("chat")
                                   .at("token_ids")
                                   .get<std::vector<int32_t>>();
    ASSERT_EQ(ids, reference_ids);
    ASSERT_FALSE(ids.empty());

    // 3. 加载每个输入位置对应的参考 logits。
    const auto T = static_cast<int64_t>(ids.size());
    const auto V = manifest.at("config").at("vocab_size").get<int64_t>();
    ASSERT_GT(V, 0);

    const auto& entry = manifest.at("reference_tensors").at("ref.chat.logits");

    ASSERT_EQ(entry.at("file"), "ref.chat.logits.bin");
    ASSERT_EQ(entry.at("dtype"), "<f4");
    ASSERT_EQ(entry.at("shape"),
              nlohmann::json(std::vector<int64_t>{T, V}));

    auto expected = base::read_f32(root / "ref.chat.logits.bin", {T, V});
    auto model = model::Qwen2Model::load(root, T);

    // 4. 连续输入 token，保留 KV Cache，并逐行比较 logits。
    for (int64_t t = 0; t < T; ++t) {
        SCOPED_TRACE("position=" + std::to_string(t));

        const auto status = model->forward_token(ids[t]);
        ASSERT_TRUE(status) << status.get_err_message();
        ASSERT_FALSE(model->failed());
        ASSERT_EQ(model->length(), t + 1);

        for (int64_t i = 0; i < model->config().num_layers; ++i)
            ASSERT_EQ(model->layer_length(i), t + 1) << "layer=" << i;

        const auto& actual = model->logits();
        ASSERT_EQ(actual.dims(), (std::vector<int64_t>{1, V}));

        const float* a = actual.ptr<float>();
        const float* e = expected.ptr<float>() + t * V;

        for (int64_t v = 0; v < V; ++v) {
            ASSERT_TRUE(std::isfinite(a[v])) << "vocab_id=" << v;

            // 沿用 short 测试的阈值，是否适用于 chat 需要实际运行验证。
            const double tolerance =
                1e-3 + 1e-4 * std::abs(double(e[v]));

            ASSERT_LE(std::abs(double(a[v]) - double(e[v])), tolerance)
                << "vocab_id=" << v
                << " actual=" << a[v]
                << " expected=" << e[v];
        }
    }
}