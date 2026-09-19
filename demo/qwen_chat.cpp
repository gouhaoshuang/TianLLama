#include "model/qwen2.h"
#include "tokenizer/qwen_tokenizer.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int32_t greedy_token(const tensor::Tensor& logits) {
    if (logits.empty() || logits.dims_size() != 2 || logits.dim(0) != 1 ||
        logits.data_type() != base::DataType::kDataTypeFp32 ||
        logits.device_type() != base::DeviceType::kDeviceCPU ||
        logits.dim(1) > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("Expected CPU FP32 logits [1, vocab_size]");
    }

    const float* scores = logits.ptr<float>();

    const int64_t vocal_size = logits.dim(1);

    int32_t best = 0;

    for (int64_t i = 0; i < vocal_size; i++) {
        if (!std::isfinite(scores[i]))
            throw std::runtime_error("Non-finite logits");
        if (scores[i] > scores[best])
            best = static_cast<int32_t>(i);
    }

    return best;
}

void forward_checked(model::Qwen2Model& model, int32_t id) {
    const auto status = model.forward_token(id);
    if (!status)
        throw std::runtime_error(status.get_err_message());
}

// decoded 是 Tokenizer 对“全部已生成 IDs”的解码结果。
// emitted 保存已经显示的文本；它的 size() 是字节数，不是汉字数。
void print_ready_text(const std::string& decoded,
                      std::string& emitted,
                      bool final = false) {
    if (decoded.size() < emitted.size() ||
        decoded.compare(0, emitted.size(), emitted) != 0) {
        throw std::runtime_error("Decoded text changed an already printed prefix");
    }

    std::size_t end = decoded.size();
    if (!final) {
        // ByteLevel 解码可能把尚未拼完整的字节显示成 U+FFFD（�）。
        // 从第一个未显示的替换字符开始暂存，等待后续 token。
        const auto pending = decoded.find("\xEF\xBF\xBD", emitted.size());
        if (pending != std::string::npos)
            end = pending;
    }

    // 当前 Tokenizer 返回合法 UTF-8；按一个 Unicode 码点的完整字节打印。
    std::size_t pos = emitted.size();
    while (pos < end) {
        const auto lead = static_cast<unsigned char>(decoded[pos]);
        const std::size_t width = lead < 0x80 ? 1 :
                                  (lead & 0xE0) == 0xC0 ? 2 :
                                  (lead & 0xF0) == 0xE0 ? 3 :
                                  (lead & 0xF8) == 0xF0 ? 4 : 0;
        if (width == 0 || width > end - pos)
            throw std::runtime_error("Unexpected UTF-8 boundary in decoded text");

        std::cout.write(decoded.data() + pos,
                        static_cast<std::streamsize>(width));
        std::cout.flush();
        pos += width;
    }
    emitted.append(decoded, emitted.size(), end - emitted.size());
}

void answer_once(model::Qwen2Model& model,
                 tokenizer::QwenTokenizer& tokenizer,
                 const std::string& question,
                 int64_t max_new_tokens) {
    if (max_new_tokens <= 0)
        throw std::invalid_argument("max_new_tokens must be positive");
    model.reset();

    const std::string prompt = tokenizer::format_qwen_prompt(question);
    const std::vector<int32_t> prompt_ids = tokenizer.encode(prompt);
    if (prompt_ids.empty())
        throw std::invalid_argument("Empty prompt IDs");

    const int64_t capacity = model.config().block.capacity;
    if (prompt_ids.size() > static_cast<std::size_t>(capacity))
        throw std::invalid_argument("Prompt exceeds KV cache capacity");

    const int64_t prompt_size = static_cast<int64_t>(prompt_ids.size());
    if (max_new_tokens > capacity - prompt_size)
        throw std::invalid_argument("Prompt plus generation budget exceeds capacity");

    for (int32_t id : prompt_ids) {
        if (id < 0 || static_cast<int64_t>(id) >= model.config().vocab_size)
            throw std::invalid_argument("Prompt ID exceeds model vocabulary");
    }

    std::cout << "正在处理问题（" << prompt_size << " 个 prompt token）..."
              << std::endl;

    // prefill
    for (int32_t id : prompt_ids)
        forward_checked(model, id);

    std::cout << "正在生成回答，最多 " << max_new_tokens
              << " 个 token。\n回答：\n"
              << std::flush;

    std::vector<int32_t> generated;
    generated.reserve(static_cast<std::size_t>(max_new_tokens));
    std::string emitted;
    std::string stop_reason = "达到生成数量上限，回答可能尚未结束";

    for (int64_t step = 0; step < max_new_tokens; ++step) {
        const int32_t next = greedy_token(model.logits());

        // 当前固定检查点的 generation_config.json 中有这两个 EOS。
        // <|im_end|> = 151645；<|endoftext|> = 151643。
        if (next == tokenizer.eos_id() || next == 151643) {
            stop_reason = "遇到结束 token";
            break;
        }

        if (!tokenizer.contains_id(next))
            throw std::runtime_error("Predicted ID is not defined by this tokenizer");

        generated.push_back(next);

        // 解码累积 IDs，不单独 decode({next})，也不重复打印全部文本。
        // print_ready_text(tokenizer.decode(generated, true), emitted);
        std::cerr << tokenizer.decode({next}, true) ;

        // 已经得到最后一个允许输出的 token，不再为它计算下一份 logits。
        if (step + 1 == max_new_tokens)
            break;

        if (model.length() >= model.config().block.capacity) {
            stop_reason = "达到 KV cache 容量上限";
            break;
        }
        forward_checked(model, next);
    }

    // 结束时不再等待后续 token：把尚未显示的解码结果原样输出。
    // print_ready_text(tokenizer.decode(generated, true), emitted, true);
    // std::cerr << "token ID: " << tokenizer.decode({next}, true) << '\n';

    std::cout << "\n回答：\n";
    if (emitted.empty())
        std::cout << "（没有生成可显示的文本）";
    std::cout << "\n[生成 " << generated.size() << " 个 token；"
              << stop_reason << "]\n";
}

} // namespace

int main(int argc, char** argv) {

    // if (argc != 2) {
    //     std::cerr << "用法：" << argv[0] << " <模型导出目录>\n";
    //     return 1;
    // }

    const std::filesystem::path root = "/data/ghs/TianLLama/test_data/qwen2_5_0_5b_model_v2";
    const int64_t capacity = 1024;

    constexpr int64_t max_new_tokens = 512;
    std::cout << "正在加载 CPU FP32 模型，请等待..." << std::endl;

    auto model = model::Qwen2Model::load(root, capacity);
    tokenizer::QwenTokenizer tokenizer((root / "tokenizer.json").string());

    std::cout << "模型已加载。输入 /exit 退出；每次提问都是独立请求。\n";

    std::string question;

    while (true) {
        std::cout << "\n问题：" << std::flush;
        if (!std::getline(std::cin, question) || question == "/exit")
            break;

        if (question.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;
        answer_once(*model, tokenizer, question, max_new_tokens);
    }

    return 0;
}