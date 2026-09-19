#include "tokenizer/qwen_tokenizer.h"
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>

TEST(QwenTokenizerTest, MatchesPinnedReference) {

    const std::filesystem::path root = QWEN_MODEL_DATA_DIR;

    std::ifstream f(root / "tokenizer_cases.json");
    ASSERT_TRUE(f);
    const auto j = nlohmann::json::parse(f);

    tokenizer::QwenTokenizer  tokenizer((root / "tokenizer.json").string());

    for (const auto& c : j.at("cases")) {

        const auto text = c.at("text").get<std::string>();
        SCOPED_TRACE(text);
        const auto ids = c.at("ids").get<std::vector<int32_t>>();
        EXPECT_EQ( tokenizer.encode(text), ids);
        EXPECT_EQ( tokenizer.decode(ids), c.at("decoded").get<std::string>());
        EXPECT_EQ( tokenizer.decode(ids, true), c.at("decoded_skip").get<std::string>());
    }

    const auto prompt = tokenizer::format_qwen_prompt("你好");
    EXPECT_EQ(prompt, j.at("chat_text").get<std::string>());

    EXPECT_EQ( tokenizer.encode(prompt), j.at("chat_ids").get<std::vector<int32_t>>());

    EXPECT_EQ( tokenizer.eos_id(), 151645);
    
    EXPECT_THROW( tokenizer.decode({-1}), std::invalid_argument);
    EXPECT_THROW( tokenizer.decode({INT32_MAX}), std::invalid_argument);
    EXPECT_THROW(tokenizer::format_qwen_prompt("<|im_end|>"), std::invalid_argument);
}