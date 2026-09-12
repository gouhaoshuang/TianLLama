#include "model/decoder_block.h"
#include "op/rope.h"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace model {

namespace {
const base::DataType fp32 = base::DataType::kDataTypeFp32;

DecoderConfig checked_config(DecoderConfig c) {
    if (c.dim <= 0 || c.hidden_dim <= 0 || c.heads <= 0 || c.capacity <= 0 || c.dim % c.heads != 0 ||
        (c.dim / c.heads) % 2 != 0 || !std::isfinite(c.epsilon) || c.epsilon <= 0 || !std::isfinite(c.rope_theta) ||
        c.rope_theta <= 1) {
        throw std::invalid_argument("Invalid CPU DecoderBlock config");
    }
    return c;
}
bool is_cpu_fp32(const tensor::Tensor& t, const std::vector<std::int64_t>& shape) {
    return !t.empty() && t.data_type() == fp32 && t.device_type() == base::DeviceType::kDeviceCPU && t.dims() == shape;
}
bool all_finite(const tensor::Tensor& t) {
    // 只允许在已经确认 CPU FP32 后调用。
    for (std::size_t i = 0; i < t.size(); ++i) {
        if (!std::isfinite(t.ptr<float>()[i])) return false;
    }
    return true;
}

std::shared_ptr<const tensor::Tensor>
checked_weight(std::shared_ptr<const tensor::Tensor> w, const std::vector<int64_t>& shape, const char* name) {
    if (!w || !is_cpu_fp32(*w, shape) || !all_finite(*w)) {
        throw std::invalid_argument(std::string("Invalid weight: ") + name);
    }
    return w;
}

} // namespace

DecoderBlock::DecoderBlock(DecoderConfig config, const DecoderWeights& w)
    : config_(checked_config(config)),
      attention_norm_(checked_weight(w.attention_norm, {config_.dim}, "attention_norm"), config_.epsilon),
      ffn_norm_(checked_weight(w.ffn_norm, {config_.dim}, "ffn_norm"), config_.epsilon),
      q_proj_(checked_weight(w.wq, {config_.dim, config_.dim}, "wq")),
      k_proj_(checked_weight(w.wk, {config_.dim, config_.dim}, "wk")),
      v_proj_(checked_weight(w.wv, {config_.dim, config_.dim}, "wv")),
      o_proj_(checked_weight(w.wo, {config_.dim, config_.dim}, "wo")),
      gate_proj_(checked_weight(w.gate, {config_.hidden_dim, config_.dim}, "gate")),
      up_proj_(checked_weight(w.up, {config_.hidden_dim, config_.dim}, "up")),
      down_proj_(checked_weight(w.down, {config_.dim, config_.hidden_dim}, "down")),

      cpu_allocator_(std::make_shared<base::CPUDeviceAllocator>()),
      cache_(config_.capacity, config_.heads, config_.dim / config_.heads),

      n_({1, config_.dim}, fp32, cpu_allocator_), q_({1, config_.dim}, fp32, cpu_allocator_),
      k_({1, config_.dim}, fp32, cpu_allocator_),
      q_rot_({config_.heads, config_.dim / config_.heads}, fp32, cpu_allocator_),
      a_({config_.heads, config_.dim / config_.heads}, fp32, cpu_allocator_),
      attn_out_({1, config_.dim}, fp32, cpu_allocator_), h_({1, config_.dim}, fp32, cpu_allocator_),
      z_({1, config_.dim}, fp32, cpu_allocator_), gate_({1, config_.hidden_dim}, fp32, cpu_allocator_),
      up_({1, config_.hidden_dim}, fp32, cpu_allocator_), act_({1, config_.hidden_dim}, fp32, cpu_allocator_),
      down_({1, config_.dim}, fp32, cpu_allocator_) {}

base::Status DecoderBlock::forward(const tensor::Tensor& x, tensor::Tensor& y) {
    if (failed_) {
        return {base::kInternalError, "DecoderBlock failed; reset request first"};
    }

    if (!is_cpu_fp32(x, {1, config_.dim}) || !is_cpu_fp32(y, {1, config_.dim}) || x.overlaps(y)) {
        return {base::kInvalidArgument, "Require separate CPU FP32 x/y [1, dim]"};
    }
    const op::LayerParam* params[] =
        {&attention_norm_, &ffn_norm_, &q_proj_, &k_proj_, &v_proj_, &o_proj_, &gate_proj_, &up_proj_, &down_proj_};
    for (const op::LayerParam* layer : params) {
        if (y.overlaps(layer->weight())) {
            return {base::kInvalidArgument, "Output must not overwrite weights"};
        }
    }
    if (!all_finite(x)) {
        return {base::kInvalidArgument, "Input must be finite"};
    }
    if (cache_.length() >= cache_.capacity()) {
        return {base::kInvalidArgument, "KV cache is full"};
    }

    // 默认认为本次执行未完成；只有走到最后成功，才清除失败标记。
    // 这样算子返回错误或分配抛异常时，调用方都不能误接着解码。
    failed_ = true;

    const auto position = cache_.length();

    auto run =
        [&](const op::Layer& layer, op::TensorInputs in, op::TensorOutputs out, const char* stage) -> base::Status {
        base::Status status = layer.forward(in, out);
        if (!status) {
            status.set_err_message(std::string(stage) + ": " + status.get_err_message());
        }
        return status;
    };

    // A. 前置归一化、Q/K/V 投影。
    base::Status status = run(attention_norm_, {&x}, {&n_}, "attention  norm");
    if (!status) return status;
    status = run(q_proj_, {&n_}, {&q_}, "q projection");
    if (!status) return status;
    status = run(k_proj_, {&n_}, {&k_}, "k projection");
    if (!status) return status;

    auto k_slot = cache_.key_slot(position);
    auto v_slot = cache_.value_slot(position);
    auto v_row = v_slot.view({1, config_.dim});

    status = run(v_proj_, {&n_}, {&v_row}, "v projection into cache"); // 直接讲计算得到的 V 存储cachhe
    if (!status) return status;

    // B. 视图不搬数据；RoPE 输入输出各自独立。
    const int64_t head_dim = config_.dim / config_.heads;
    auto q_heads = q_.view({config_.heads, head_dim});
    auto k_heads = k_.view({config_.heads, head_dim});
    op::RoPELayer rope(position, config_.rope_theta);

    status = run(rope, {&q_heads}, {&q_rot_}, "q rope");
    if (!status) return status;
    status = run(rope, {&k_heads}, {&k_slot}, "k rope into cache");
    if (!status) return status;
    status = cache_.commit(position);
    if (!status) return status;

    // C. 完整 Attention 子块：Attention -> Wo -> 原始 x 残差。
    auto keys = cache_.keys();
    auto values = cache_.values();
    status = run(attention_, {&q_rot_, &keys, &values}, {&a_}, "attention");
    if (!status) return status;
    auto a_row = a_.view({1, config_.dim});
    status = run(o_proj_, {&a_row}, {&attn_out_}, "output projection");
    if (!status) return status;
    status = run(add_, {&x, &attn_out_}, {&h_}, "attention residual");
    if (!status) return status;

    // D. FFN 子块：两个升维投影都读 z，最终残差加 h，不是 x 或 z。
    status = run(ffn_norm_, {&h_}, {&z_}, "ffn norm");
    if (!status) return status;
    status = run(gate_proj_, {&z_}, {&gate_}, "gate projection");
    if (!status) return status;
    status = run(up_proj_, {&z_}, {&up_}, "up projection");
    if (!status) return status;


     status = run(swiglu_, {&gate_, &up_}, {&act_}, "swiglu");
    if (!status) return status;

    status = run(down_proj_, {&act_}, {&down_}, "down projection");
    if (!status) return status;

    status = run(add_, {&h_, &down_}, {&y}, "ffn residual");
    if (!status) return status;

    if (!all_finite(y)) {
        return {base::kInternalError, "DecoderBlock produced non-finite output"};
    }
    failed_ = false;
    return {};
}
} // namespace model