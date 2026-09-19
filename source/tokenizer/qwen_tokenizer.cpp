#include "tokenizer/qwen_tokenizer.h"
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <tokenizers_cpp.h>
#include <unordered_set>

namespace tokenizer {

using json = nlohmann::json;

struct QwenTokenizer::Impl {
    std::unique_ptr<tokenizers::Tokenizer> backend;
    std::unordered_set<int32_t> valid_ids;
    std::unordered_set<int32_t> special_ids;

    int32_t eos = -1;
};

/*
std::istreambuf_iterator<char>：流缓冲区迭代器。
    它是未格式化的（unformatted），不会像 std::istream_iterator 那样自动跳过空格和换行符，
    而是按字节直接读取底层缓冲区，速度非常快。
std::istreambuf_iterator<char>(file) 代表文件的起始位置。
std::istreambuf_iterator<char>()（不传参）是一个默认构造的迭代器，
    在 C++ 中用作流的结束标志（End-of-Stream / EOF）。

std::string 的范围构造函数：std::string(begin_iterator, end_iterator)，
    它会读取从起点到终点的所有字符，构造成一个完整的字符串。

    最令人抓狂的语法细节——额外的括号 (( ... ))：
    注意 (std::istreambuf_iterator<char>(file)) 外面套了一层括号。
    这是为了防止 C++ 中臭名昭著的 “Most Vexing Parse”（最烦人的语法分析）。

    如果不加这层括号写成 std::string blob(std::istreambuf_iterator<char>(file), ...)，
    编译器会把它误认为是一个函数的声明
    （声明了一个返回值为 std::string、名为 blob、参数为迭代器的函数），而不是在定义一个变量。
    加了括号后，编译器才能正确识别这是在调用构造函数初始化对象。

blob：Binary Large Object（二进制大对象）的缩写，常用来指代一块未经处理的原始数据内存。
*/
QwenTokenizer::QwenTokenizer(const std::string& path)
    : impl_(std::make_unique<Impl>()) {

    std::ifstream file(path, std::ios::binary);

    if (!file) throw std::runtime_error("Cannot open tokenizer: " + path);

    /*

    将整个文件内容以极高的效率一次性拷贝到名为 blob 的 std::string 中。
    */
    const std::string blob((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

    const auto j = json::parse(blob);

    if (j.at("model").at("type") != "BPE" ||       // 检查分词核心模型是否是 BPE（Byte-Pair Encoding，字节对编码）。
        j.at("normalizer").at("type") != "NFC" ||  // 检查文本正规化（Normalization）方式是否是 NFC（Normalization Form C，规范对齐格式）。
        j.at("decoder").at("type") != "ByteLevel") // 查解码器（Decoder）是否是 ByteLevel（字节级别）。
        throw std::invalid_argument("Expected the pinned Qwen2 tokenizer.json");

    auto add_id = [&](const json& v) {
        const auto n = v.get<int64_t>();
        if (n < 0 || n > INT32_MAX)
            throw std::invalid_argument("Invalid tokenizer ID");

        impl_->valid_ids.insert(static_cast<int32_t>(n));
        return static_cast<int32_t>(n);
    };

    for (const auto& item : j.at("model").at("vocab").items())
        add_id(item.value());

    for (const auto& item : j.at("added_tokens")) {
        const int32_t id = add_id(item.at("id"));

        if (item.at("special").get<bool>())
            impl_->special_ids.insert(id);

        if (item.at("content") == "<|im_end|>")
            impl_->eos = id;
    }

    if (impl_->eos != 151645 || impl_->valid_ids.empty())
        throw std::invalid_argument("Tokenizer does not match this Qwen checkpoint");

    // 完整 JSON 包含 normalizer、pre_tokenizer、merges、added_tokens。
    impl_->backend = tokenizers::Tokenizer::FromBlobJSON(blob);
    if (!impl_->backend) throw std::runtime_error("Tokenizer construction failed");
}

QwenTokenizer::~QwenTokenizer() = default;

bool QwenTokenizer::contains_id(int32_t id) const {
    return impl_->valid_ids.contains(id);
}
int32_t QwenTokenizer::eos_id() const { return impl_->eos; }

std::vector<int32_t> QwenTokenizer::encode(const std::string& text) {
    // dump 的 strict UTF-8 校验会在进入 Rust FFI 前拒绝非法字节串。
    const auto checked_utf8 = json(text).dump();
    // dump() 默认严格检查 UTF-8。如果字符串包含非法 UTF-8 字节，
    //  就会抛出 nlohmann::json::type_error，
    //  后面的代码不会继续执行，除非外层捕获并处理异常。

    (void)checked_utf8;

    if (text.empty()) return {};

    auto ids = impl_->backend->Encode(text);

    for (auto id : ids)
        if (!contains_id(id))
            throw std::runtime_error("Unknown encoded ID");
    return ids;
}

std::string QwenTokenizer::decode(const std::vector<int32_t>& ids, bool skip) {
    std::vector<int32_t> kept;

    kept.reserve(ids.size());

    for (auto id : ids) {
        if (!contains_id(id))
            throw std::invalid_argument("Unknown decode ID");
        if (!skip || !impl_->special_ids.contains(id))
            kept.push_back(id);
    }

    if (kept.empty()) return {};

    return impl_->backend->Decode(kept);
}

std::string format_qwen_prompt(const std::string& user, const std::string& system) {
    // 本课只实现一个 system + 一个 user，不处理工具或任意 Jinja 模板。
    for (const auto* text : {&user, &system}) {
        for (const auto* marker : {"<|im_start|>", "<|im_end|>", "<|endoftext|>"})
            if (text->find(marker) != std::string::npos)
                throw std::invalid_argument("Chat content contains a control marker");
    }
    return "<|im_start|>system\n" + system + "<|im_end|>\n" +
           "<|im_start|>user\n" + user + "<|im_end|>\n" +
           "<|im_start|>assistant\n";
}
} // namespace tokenizer

