#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tokenizer {

class QwenTokenizer {

  public:
    explicit QwenTokenizer(const std::string& tokenizer_json);
    ~QwenTokenizer();

    QwenTokenizer(const QwenTokenizer& other) = delete;
    QwenTokenizer& operator=(const QwenTokenizer& other) = delete;

    std::vector<int32_t> encode(const std::string& text);

    std::string decode(const std::vector<int32_t>& ids,
                                bool skip_special_tokens = false);

    bool contains_id(int32_t id) const;
    int32_t eos_id() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string format_qwen_prompt(
    const std::string& user,
    const std::string& system =
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.");



} // namespace tokenizer
