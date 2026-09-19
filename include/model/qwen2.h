#pragma once
#include "base/kv_cache.h"
#include "op/add.h"
#include "op/attention.h"
#include "op/linear.h"
#include "op/rmsnorm.h"
#include "op/swiglu.h"
#include <cstdint>
#include <memory>

namespace model {

using Qwen2Weight = std::shared_ptr<const tensor::Tensor>;

struct Qwen2Config {
    int64_t dim = 0;
    int64_t hidden_dim = 0;
    int64_t q_heads = 0;
    int64_t kv_heads = 0;
    int64_t head_dim = 0; // 明确提供，不把 dim/q_heads 写进每个组件
    int64_t capacity = 128;
    float epsilon = 1e-6F;
    double rope_theta = 1000000.0;

    int64_t q_dim() const { return q_heads * head_dim; }
    int64_t kv_dim() const { return kv_heads * head_dim; }
};

struct Qwen2AttentionWeights {
    Qwen2Weight wq, wk, wv, wo;
    Qwen2Weight bq, bk, bv; // Qwen2 架构要求三者存在
};
struct Qwen2LayerWeights {
    Qwen2Weight attention_norm, ffn_norm;
    Qwen2AttentionWeights attention;
    Qwen2Weight gate, up, down;
};

class Qwen2Attention {

  public:
    Qwen2Attention(Qwen2Config config, const Qwen2AttentionWeights& weights);
    base::Status forward(const tensor::Tensor& normalized_x, tensor::Tensor& out);

    int64_t length() const { return cache_.length(); }
    bool failed() const { return failed_; }
    void reset() {
        cache_.reset();
        failed_ = false;
    }

  private:
    Qwen2Config config_;
    Qwen2AttentionWeights weights_;

    op::LinearLayer q_proj_, k_proj_, v_proj_, o_proj_;
    op::AttentionLayer  attention_;
    std::shared_ptr<base::CPUDeviceAllocator> cpu_allocator_;

    base::KVCache cache_;
    tensor::Tensor q_, k_, q_rot_, a_;
    bool failed_ = false;
};

class Qwen2DecoderLayer {

  public:
    Qwen2DecoderLayer(Qwen2Config config, const Qwen2LayerWeights& weights);
    base::Status forward(const tensor::Tensor& x, tensor::Tensor& y);
    
    int64_t length() const { return attention_.length(); }
    bool failed() const { return failed_ || attention_.failed(); }
    void reset() {
        attention_.reset();
        failed_ = false;
    }

  private:
    Qwen2Config config_;
    Qwen2LayerWeights weights_;

    op::RmsNormLayer attention_norm_, ffn_norm_;
    Qwen2Attention attention_;
    op::LinearLayer gate_proj_, up_proj_, down_proj_;
    op::SwiGLULayer swiglu_;
    op::AddLayer add_;

    std::shared_ptr<base::CPUDeviceAllocator> cpu_allocator_;
    tensor::Tensor n_, attn_out_, h_, z_, gate_, up_, act_, down_;
    bool failed_ = false;
};

} // namespace model
