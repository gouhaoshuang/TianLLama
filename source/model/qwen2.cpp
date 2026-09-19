#include "model/qwen2.h"
#include "op/rope.h"
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace model {

constexpr auto fp32 = base::DataType::kDataTypeFp32;

namespace {
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

void require_weight(const Qwen2Weight& w,
                    const std::vector<int64_t>& shape,
                    const char* name) {
    if (!w || !is_cpu_fp32(*w, shape) || !all_finite(*w))
        throw std::invalid_argument(std::string("Invalid Qwen2 weight: ") + name);
}

Qwen2AttentionWeights checked_attention_weights(
    const Qwen2Config& c,
    const Qwen2AttentionWeights& w) {
    require_weight(w.wq, {c.q_dim(), c.dim}, "wq");
    require_weight(w.wk, {c.kv_dim(), c.dim}, "wk");
    require_weight(w.wv, {c.kv_dim(), c.dim}, "wv");
    require_weight(w.wo, {c.dim, c.q_dim()}, "wo");
    require_weight(w.bq, {c.q_dim()}, "bq");
    require_weight(w.bk, {c.kv_dim()}, "bk");
    require_weight(w.bv, {c.kv_dim()}, "bv");
    return w;
}

Qwen2LayerWeights checked_layer_weights(
    const Qwen2Config& c,
    const Qwen2LayerWeights& w) {
    require_weight(w.attention_norm, {c.dim}, "attention_norm");
    require_weight(w.ffn_norm, {c.dim}, "ffn_norm");
    require_weight(w.gate, {c.hidden_dim, c.dim}, "gate");
    require_weight(w.up, {c.hidden_dim, c.dim}, "up");
    require_weight(w.down, {c.dim, c.hidden_dim}, "down");
    // attention 的全部参数由随后构造的 Qwen2Attention 校验。
    return w;
}

bool overlaps_any(const tensor::Tensor& y,
                  std::initializer_list<Qwen2Weight> weights) {
    for (const auto& w : weights)
        if (w && y.overlaps(*w)) return true;
    return false;
}

bool overlaps_attention(const tensor::Tensor& y, const Qwen2AttentionWeights& w) {
    return overlaps_any(y, {w.wq, w.wk, w.wv, w.wo, w.bq, w.bk, w.bv});
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



} // namespace model
