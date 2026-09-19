#include "model/qwen2.h"
#include "base/utils.h"
#include "op/rope.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace model {

constexpr auto fp32 = base::DataType::kDataTypeFp32;

namespace {
QwenModelConfig check_model_config(QwenModelConfig c) {
    const auto& block_config = c.block;
    if (c.num_layers <= 0 || c.vocab_size <= 0 || c.vocab_size > INT32_MAX ||
        block_config.dim <= 0 || block_config.hidden_dim <= 0 || block_config.q_heads <= 0 ||
        block_config.kv_heads <= 0 || block_config.head_dim <= 0 || block_config.capacity <= 0 ||
        block_config.dim % block_config.q_heads != 0 || block_config.dim / block_config.q_heads != block_config.head_dim ||
        block_config.q_heads % block_config.kv_heads != 0 || block_config.head_dim % 2 != 0 ||
        !std::isfinite(block_config.epsilon) || block_config.epsilon <= 0 ||
        !std::isfinite(block_config.rope_theta) || block_config.rope_theta <= 1)
        throw std::invalid_argument("Invalid  Qwen2Model config");
    return c;
}

void require_model_weight(const Qwen2Weight& weights,
                          const std::vector<int64_t>& shape) {
    if (!weights || weights->empty() || weights->dims() != shape ||
        weights->device_type() != base::DeviceType::kDeviceCPU ||
        weights->data_type() != base::DataType::kDataTypeFp32)
        throw std::invalid_argument("Invalid model weight");
    for (std::size_t i = 0; i < weights->size(); ++i)
        if (!std::isfinite(weights->ptr<float>()[i]))
            throw std::invalid_argument("Non-finite model weight");
}

QwenModelWeights check_model_weights(const QwenModelConfig& c,
                                     QwenModelWeights weights) {
    if (weights.layers.size() != static_cast<std::size_t>(c.num_layers))
        throw std::invalid_argument("Layer count mismatch");
    require_model_weight(weights.embedding, {c.vocab_size, c.block.dim});
    require_model_weight(weights.final_norm, {c.block.dim});
    if (weights.lm_head != weights.embedding)
        throw std::invalid_argument("This model requires tied embedding/lm_head");
    // 各层形状和有限值由 Qwen2DecoderLayer 构造函数继续检查。
    return weights;
}

Qwen2Config checked_config(Qwen2Config c) {
    if (c.dim <= 0 || c.hidden_dim <= 0 || c.q_heads <= 0 ||
        c.kv_heads <= 0 || c.head_dim <= 0 || c.capacity <= 0 ||
        c.q_heads % c.kv_heads != 0 || c.head_dim % 2 != 0 ||
        !std::isfinite(c.epsilon) || c.epsilon <= 0 ||
        !std::isfinite(c.rope_theta) || c.rope_theta <= 1) {
        throw std::invalid_argument("Invalid Qwen2 CPU config");
    }
    const auto max = std::numeric_limits<int64_t>::max();
    if (c.q_heads > max / c.head_dim || c.kv_heads > max / c.head_dim)
        throw std::overflow_error("Qwen2 projection dimension overflow");
    return c;
}
bool is_cpu_fp32(const tensor::Tensor& t, const std::vector<int64_t>& shape) {
    return !t.empty() && t.device_type() == base::DeviceType::kDeviceCPU &&
           t.data_type() == fp32 && t.dims() == shape;
}

bool all_finite(const tensor::Tensor& t) {
    // 仅在已确认 CPU FP32 后调用。
    for (std::size_t i = 0; i < t.size(); ++i)
        if (!std::isfinite(t.ptr<float>()[i])) return false;
    return true;
}

void require_weight(const Qwen2Weight& weights,
                    const std::vector<int64_t>& shape,
                    const char* name) {
    if (!weights || !is_cpu_fp32(*weights, shape) || !all_finite(*weights))
        throw std::invalid_argument(std::string("Invalid Qwen2 weight: ") + name);
}

Qwen2AttentionWeights checked_attention_weights(
    const Qwen2Config& c,
    const Qwen2AttentionWeights& weights) {
    require_weight(weights.wq, {c.q_dim(), c.dim}, "wq");
    require_weight(weights.wk, {c.kv_dim(), c.dim}, "wk");
    require_weight(weights.wv, {c.kv_dim(), c.dim}, "wv");
    require_weight(weights.wo, {c.dim, c.q_dim()}, "wo");
    require_weight(weights.bq, {c.q_dim()}, "bq");
    require_weight(weights.bk, {c.kv_dim()}, "bk");
    require_weight(weights.bv, {c.kv_dim()}, "bv");
    return weights;
}

Qwen2LayerWeights checked_layer_weights(
    const Qwen2Config& c,
    const Qwen2LayerWeights& weights) {
    require_weight(weights.attention_norm, {c.dim}, "attention_norm");
    require_weight(weights.ffn_norm, {c.dim}, "ffn_norm");
    require_weight(weights.gate, {c.hidden_dim, c.dim}, "gate");
    require_weight(weights.up, {c.hidden_dim, c.dim}, "up");
    require_weight(weights.down, {c.dim, c.hidden_dim}, "down");
    // attention 的全部参数由随后构造的 Qwen2Attention 校验。
    return weights;
}

bool overlaps_any(const tensor::Tensor& y,
                  std::initializer_list<Qwen2Weight> weights) {
    for (const auto& weights : weights)
        if (weights && y.overlaps(*weights)) return true;
    return false;
}

bool overlaps_attention(const tensor::Tensor& y, const Qwen2AttentionWeights& weights) {
    return overlaps_any(y, {weights.wq, weights.wk, weights.wv, weights.wo, weights.bq, weights.bk, weights.bv});
}

base::Status run(const op::Layer& layer, op::TensorInputs in, op::TensorOutputs out, const char* stage) {
    auto status = layer.forward(in, out);
    if (!status)
        status.set_err_message(std::string(stage) + ": " + status.get_err_message());
    return status;
}

} // namespace

Qwen2Attention::Qwen2Attention(Qwen2Config config, const Qwen2AttentionWeights& weights)
    : config_(checked_config(config)),
      weights_(checked_attention_weights(config_, weights)),
      q_proj_(weights_.wq, weights_.bq),
      k_proj_(weights_.wk, weights_.bk),
      v_proj_(weights_.wv, weights_.bv),
      o_proj_(weights_.wo),

      cpu_allocator_(std::make_shared<base::CPUDeviceAllocator>()),
      cache_(config_.capacity, config_.kv_heads, config_.head_dim),

      q_({1, config_.q_dim()}, fp32, cpu_allocator_),
      k_({1, config_.kv_dim()}, fp32, cpu_allocator_),
      q_rot_({config_.q_heads, config_.head_dim}, fp32, cpu_allocator_),
      a_({config_.q_heads, config_.head_dim}, fp32, cpu_allocator_) {}

base::Status Qwen2Attention::forward(
    const tensor::Tensor& n,
    tensor::Tensor& out) {
    if (failed_)
        return {base::kInternalError, "Qwen2Attention failed; reset first"};
    if (!is_cpu_fp32(n, {1, config_.dim}) ||
        !is_cpu_fp32(out, {1, config_.dim}) || out.overlaps(n) ||
        overlaps_attention(out, weights_))
        return {base::kInvalidArgument, "Invalid Qwen2Attention input/output"};
    if (!all_finite(n))
        return {base::kInvalidArgument, "Qwen2Attention input must be finite"};
    if (length() >= config_.capacity)
        return {base::kInvalidArgument, "Qwen2Attention cache is full"};

    failed_ = true; // 从此处开始，错误/异常都要求 reset。
    const int64_t position = length();
    auto status = run(q_proj_, {&n}, {&q_}, "q projection");
    if (!status) return status;
    status = run(k_proj_, {&n}, {&k_}, "k projection");
    if (!status) return status;

    auto k_slot = cache_.key_slot(position);
    auto v_slot = cache_.value_slot(position);
    auto v_row = v_slot.view({1, config_.kv_dim()});
    status = run(v_proj_, {&n}, {&v_row}, "v projection");
    if (!status) return status;

    auto q_heads = q_.view({config_.q_heads, config_.head_dim});
    auto k_heads = k_.view({config_.kv_heads, config_.head_dim});
    op::RoPELayer rope(position, config_.rope_theta, base::RopeLayout::kHalfSplit);
    status = run(rope, {&q_heads}, {&q_rot_}, "q rope");
    if (!status) return status;
    status = run(rope, {&k_heads}, {&k_slot}, "k rope");
    if (!status) return status;
    status = cache_.commit(position);
    if (!status) return status;

    auto keys = cache_.keys();
    auto values = cache_.values();
    status = run(attention_, {&q_rot_, &keys, &values}, {&a_}, "gqa");
    if (!status) return status;

    auto a_row = a_.view({1, config_.q_dim()});
    status = run(o_proj_, {&a_row}, {&out}, "o projection");
    if (!status) return status;

    if (!all_finite(out))
        return {base::kInternalError, "Qwen2Attention produced non-finite output"};
    failed_ = false;
    return {};
}

Qwen2DecoderLayer::Qwen2DecoderLayer(
    Qwen2Config config,
    const Qwen2LayerWeights& weights)
    : config_(checked_config(config)),
      weights_(checked_layer_weights(config_, weights)),

      attention_norm_(weights_.attention_norm, config_.epsilon),
      ffn_norm_(weights_.ffn_norm, config_.epsilon),
      attention_(config_, weights_.attention),
      gate_proj_(weights_.gate), up_proj_(weights_.up), down_proj_(weights_.down),
      cpu_allocator_(std::make_shared<base::CPUDeviceAllocator>()),

      n_({1, config_.dim}, fp32, cpu_allocator_),
      attn_out_({1, config_.dim}, fp32, cpu_allocator_),
      h_({1, config_.dim}, fp32, cpu_allocator_),
      z_({1, config_.dim}, fp32, cpu_allocator_),
      gate_({1, config_.hidden_dim}, fp32, cpu_allocator_),
      up_({1, config_.hidden_dim}, fp32, cpu_allocator_),
      act_({1, config_.hidden_dim}, fp32, cpu_allocator_),
      down_({1, config_.dim}, fp32, cpu_allocator_) {}

base::Status Qwen2DecoderLayer::forward(
    const tensor::Tensor& x,
    tensor::Tensor& y) {
    if (failed())
        return {base::kInternalError, "Qwen2DecoderLayer failed; reset first"};
    if (!is_cpu_fp32(x, {1, config_.dim}) || !is_cpu_fp32(y, {1, config_.dim}) ||
        y.overlaps(x) || overlaps_attention(y, weights_.attention) ||
        overlaps_any(y, {weights_.attention_norm, weights_.ffn_norm, weights_.gate, weights_.up, weights_.down}))
        return {base::kInvalidArgument, "Invalid Qwen2DecoderLayer input/output"};
    if (!all_finite(x))
        return {base::kInvalidArgument, "Qwen2DecoderLayer input must be finite"};
    if (length() >= config_.capacity)
        return {base::kInvalidArgument, "Qwen2DecoderLayer cache is full"};

    failed_ = true;
    auto status = run(attention_norm_, {&x}, {&n_}, "attention norm");
    if (!status) return status;
    status = attention_.forward(n_, attn_out_);
    if (!status) return status;

    status = run(add_, {&x, &attn_out_}, {&h_}, "attention residual");
    if (!status) return status;

    status = run(ffn_norm_, {&h_}, {&z_}, "ffn norm");
    if (!status) return status;

    status = run(gate_proj_, {&z_}, {&gate_}, "gate");
    if (!status) return status;

    status = run(up_proj_, {&z_}, {&up_}, "up");
    if (!status) return status;

    status = run(swiglu_, {&gate_, &up_}, {&act_}, "swiglu");
    if (!status) return status;

    status = run(down_proj_, {&act_}, {&down_}, "down");
    if (!status) return status;

    status = run(add_, {&h_, &down_}, {&y}, "ffn residual");
    if (!status) return status;

    if (!all_finite(y))
        return {base::kInternalError, "Qwen2DecoderLayer produced non-finite output"};
    failed_ = false;
    return {};
    /**
     *
    n = attention_norm(x)
    attn_out = Qwen2Attention(n)
    h = x + attn_out
    z = ffn_norm(h)
    y = h + down(SwiGLU(gate(z), up(z)))

     两个常见连接错误：

    - 第一次残差要加原始 x，不是归一化后的 n。
    - 第二次残差要加 h，不是 x 或 z。
     */
}

Qwen2Model::Qwen2Model(QwenModelConfig config, QwenModelWeights weights)
    : config_(check_model_config(config)),
      weights_(check_model_weights(config_, std::move(weights))),

      cpu_allocator_(std::make_shared<base::CPUDeviceAllocator>()),

      final_norm_(weights_.final_norm, config_.block.epsilon),
      lm_head_(weights_.lm_head),

      hidden_a_({1, config_.block.dim}, base::DataType::kDataTypeFp32, cpu_allocator_),
      hidden_b_({1, config_.block.dim}, base::DataType::kDataTypeFp32, cpu_allocator_),

      logits_({1, config_.vocab_size}, base::DataType::kDataTypeFp32, cpu_allocator_) {

    layers_.reserve(weights_.layers.size());

    for (const auto& weights : weights_.layers)
        layers_.push_back(std::make_unique<Qwen2DecoderLayer>(config_.block, weights));
}

base::Status Qwen2Model::forward_token(int32_t id) {

    logits_valid_ = false;

    if (failed_) {
        return {base::kInternalError, "QwenModel failed; reset the whole request"};
    }
    if (id < 0 || static_cast<int64_t>(id) >= config_.vocab_size)
        return {base::kInvalidArgument, "Token ID out of embedding range"};
    if (length_ >= config_.block.capacity)
        return {base::kInvalidArgument, "QwenModel cache is full"};
    for (const auto& layer : layers_) {
        if (layer->failed() || layer->length() != length_) {
            failed_ = true;
            return {base::kInternalError, "Layer cache positions disagree"};
        }
    }

    // 从这里开始可能修改部分层缓存；任何错误或异常都必须整请求 reset。
    failed_ = true;

    const size_t dim = static_cast<std::size_t>(config_.block.dim);
    const size_t offset = static_cast<size_t>(id) * dim;

    // 词表映射
    std::copy_n(weights_.embedding->ptr<float>() + offset, dim, hidden_a_.ptr<float>());

    tensor::Tensor* current = &hidden_a_;
    tensor::Tensor* next = &hidden_b_;

    for (size_t i = 0; i < layers_.size(); i++) {
        base::Status status = layers_[i]->forward(*current, *next);
        if (!status) {
            status.set_err_message("layer " + std::to_string(i) + ": " +
                                   status.get_err_message());
            return status;
        }
        if (layers_[i]->length() != length_ + 1)
            return {base::kInternalError, "Unexpected layer cache advance"};
        std::swap(current, next);
    }
    auto status = final_norm_.forward({current}, {next});
    if (!status) return status;

    status = lm_head_.forward({next}, {&logits_});
    if (!status) return status;

    for (std::size_t i = 0; i < logits_.size(); ++i)
        if (!std::isfinite(logits_.ptr<float>()[i]))
            return {base::kInternalError, "Non-finite final logits"};

    ++length_;
    failed_ = false;
    logits_valid_ = true;
    return {};
}

const tensor::Tensor& Qwen2Model::logits() const {
    if (!logits_valid_)
        throw std::logic_error("No valid logits; call forward_token successfully first");
    return logits_;
}

void Qwen2Model::reset() {
    for (auto& layer : layers_)
        layer->reset();
    length_ = 0;
    failed_ = false;
    logits_valid_ = false;
}

using json = nlohmann::json;
std::unique_ptr<Qwen2Model> Qwen2Model::load(
    const std::filesystem::path& root,
    int64_t capacity) {

    std::ifstream manifest(root / "manifest.json");
    if (!manifest) throw std::runtime_error("Missing completed manifest.json");
    const auto j = json::parse(manifest);
    const auto& c = j.at("config");

    if (j.at("format_version") != 2 ||
        j.at("model_id") != "Qwen/Qwen2.5-0.5B-Instruct" ||
        j.at("compute_dtype") != "float32" ||
        j.at("weight_layout") != "out_in" ||
        j.at("rope_layout") != "half_split" ||
        j.at("lm_head_alias") != "embedding" ||
        c.at("tie_word_embeddings") != true) {
        throw std::invalid_argument("Unsupported model manifest");
    }

    QwenModelConfig config;
    Qwen2Config& block_config = config.block;

    block_config.dim = c.at("hidden_size").get<int64_t>();
    block_config.hidden_dim = c.at("intermediate_size").get<int64_t>();
    block_config.q_heads = c.at("num_attention_heads").get<int64_t>();
    block_config.kv_heads = c.at("num_key_value_heads").get<int64_t>();

    if (block_config.q_heads <= 0 ||
        block_config.dim <= 0 ||
        block_config.dim % block_config.q_heads != 0) {
        throw std::invalid_argument("Invalid hidden_size/num_attention_heads");
    }

    block_config.head_dim = block_config.dim / block_config.q_heads;
    block_config.capacity = capacity;
    block_config.epsilon = c.at("rms_norm_eps").get<float>();
    block_config.rope_theta = c.at("rope_theta").get<double>();
    config.num_layers = c.at("num_hidden_layers").get<int64_t>();
    config.vocab_size = c.at("vocab_size").get<int64_t>();

    const auto& weights_info = j.at("weights_file");
    const std::filesystem::path weights_path = root / "weights.bin";
    const auto& total_bytes = weights_info.at("nbytes");

    const auto& index = j.at("tensors");

    auto load_weight = [&](const std::string& name,
                           const std::vector<int64_t>& expected) -> Qwen2Weight {
        const auto& entry = index.at(name); // 缺少该名字会抛异常。
        if (entry.at("dtype") != "<f4" || entry.at("shape") != json(expected))
            throw std::invalid_argument("Weight dtype/shape mismatch: " + name);

        // expected 来自本函数已验证的固定模型尺寸，不来自任意 JSON shape。
        std::uint64_t bytes = 4;

        for (const auto dim : expected)
            bytes *= static_cast<std::uint64_t>(dim);

        const auto& offset = entry.at("offset_bytes");
        const auto& nbytes = entry.at("nbytes");
        if (!offset.is_number_unsigned() || !nbytes.is_number_unsigned() ||
            nbytes.get<std::uint64_t>() != bytes)
            throw std::invalid_argument("Invalid weight offset/nbytes: " + name);

        // 对齐、文件边界、短读、NaN/Inf 检查由现有 utils 完成。
        return std::make_shared<tensor::Tensor>(
            base::read_f32(weights_path, expected, offset.get<std::uint64_t>()));
    };

    // 3. 按现有权重结构赋值，24 层只是重复同一组加载操作。
    QwenModelWeights weights;
    weights.embedding = load_weight("embedding", {config.vocab_size, block_config.dim});
    weights.final_norm = load_weight("final_norm", {block_config.dim});
    weights.lm_head = weights.embedding;

    weights.layers.reserve(static_cast<std::size_t>(config.num_layers));
    for (size_t i = 0; i < config.num_layers; i++) {
        const std::string p = "layers." + std::to_string(i) + ".";

        Qwen2LayerWeights layer;
        layer.attention_norm = load_weight(p + "attention_norm", {block_config.dim});
        layer.ffn_norm = load_weight(p + "ffn_norm", {block_config.dim});
        layer.gate = load_weight(p + "gate", {block_config.hidden_dim, block_config.dim});
        layer.up = load_weight(p + "up", {block_config.hidden_dim, block_config.dim});
        layer.down = load_weight(p + "down", {block_config.dim, block_config.hidden_dim});

        Qwen2AttentionWeights& attention = layer.attention;
        attention.wq = load_weight(p + "wq", {block_config.q_dim(), block_config.dim});
        attention.wk = load_weight(p + "wk", {block_config.kv_dim(), block_config.dim});
        attention.wv = load_weight(p + "wv", {block_config.kv_dim(), block_config.dim});
        attention.wo = load_weight(p + "wo", {block_config.dim, block_config.q_dim()});
        attention.bq = load_weight(p + "bq", {block_config.q_dim()});
        attention.bk = load_weight(p + "bk", {block_config.kv_dim()});
        attention.bv = load_weight(p + "bv", {block_config.kv_dim()});

        weights.layers.push_back(std::move(layer));
    }
    // 4. 权重准备好后，交给第七步的构造函数。
    return std::make_unique<Qwen2Model>(config, std::move(weights));
}

} // namespace model
