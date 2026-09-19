#pragma once
#include "base/kv_cache.h"
#include "op/add.h"
#include "op/attention.h"
#include "op/linear.h"
#include "op/rmsnorm.h"
#include "op/swiglu.h"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

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

    int64_t num_layers = 0;
    int64_t vocab_size = 0;

    int64_t q_dim() const { return q_heads * head_dim; }
    int64_t kv_dim() const { return kv_heads * head_dim; }
};

struct QwenModelConfig {
    Qwen2Config block;
    int64_t num_layers = 0;
    int64_t vocab_size = 0;
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

struct QwenModelWeights {
    Qwen2Weight embedding;
    Qwen2Weight final_norm;
    Qwen2Weight lm_head; // 本课必须指向 embedding 的同一个 Tensor。
    std::vector<Qwen2LayerWeights> layers;
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
    op::AttentionLayer attention_;
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

class  Qwen2Model {

  public:
     Qwen2Model(QwenModelConfig config, QwenModelWeights weights);

     Qwen2Model(const  Qwen2Model&) = delete;
     Qwen2Model& operator=(const  Qwen2Model&) = delete;

    static std::unique_ptr< Qwen2Model> load(
        const std::filesystem::path& root,
        int64_t capacity = 1024);

    base::Status forward_token(int32_t token_id);

    const tensor::Tensor& logits() const;
    void reset();
    bool failed() const { return failed_; }
    int64_t length() const { return length_; }
    int64_t layer_length(std::size_t i) const { return layers_.at(i)->length(); }
    const QwenModelConfig& config() const { return config_; }

  private:
    QwenModelConfig config_;
    QwenModelWeights weights_;

    std::shared_ptr<base::CPUDeviceAllocator> cpu_allocator_;
    std::vector<std::unique_ptr<Qwen2DecoderLayer>> layers_;

    op::RmsNormLayer final_norm_;
    op::LinearLayer lm_head_;

    tensor::Tensor hidden_a_, hidden_b_, logits_;

    int64_t length_ = 0;
    bool failed_ = false;
    bool logits_valid_ = false;
};

} // namespace model
