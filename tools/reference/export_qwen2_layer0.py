import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen2 import modeling_qwen2 as qwen2
from tools.utils.utils import load_model
"""
cd /data/ghs/TianLLama
python -m tools.reference.export_qwen2_layer0 \
    --out /data/ghs/TianLLama/test_data/qwen2_5_0_5b_layer0_v1
"""

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

@torch.inference_mode()
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"不覆盖已有数据，请换一个输出目录：{args.out}")

    if transformers.__version__ != "4.57.6":
        raise RuntimeError("本教程要求 agent 中的 transformers==4.57.6")

    torch.set_num_threads(1)
    torch.manual_seed(0)

    print("加载", MODEL_ID, REVISION, flush=True)
    tokenizer, model = load_model(MODEL_ID, REVISION)

    config = model.config

    assert (
        config.hidden_size,
        config.intermediate_size,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.num_hidden_layers,
    ) == (896, 4864, 14, 2, 24)

    assert config.rms_norm_eps == 1e-6 and config.rope_theta == 1000000
    assert not config.use_sliding_window

    block = model.model.layers[0]
    assert all(p.dtype == torch.float32 for p in block.parameters())

    prompt = "你好，请介绍一下你自己。"
    # input_ids: (batch_size, sequence_length)
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][
        :, :8
    ]
    T = ids.shape[1]  # 获取当前 token 序列的实际长度（Sequence Length）。
    assert 3 <= T <= 8
    pos = torch.arange(T).unsqueeze(0) #  (1, T)

    """
    torch.arange(T)：生成一个从 0 到 T-1 的一维整数序列，代表每个 token 在句子中的绝对位置（例如T = 5 时，生成 [0, 1, 2, 3, 4]），形状为 (T,)。
    .unsqueeze(0)：在第 0 维增加一个“批次（Batch）”维度，使形状变为 (1, T)。
    """
    x = model.model.embed_tokens(ids)
    """
    输入 ids 的形状是 (batch_size, T)。
    输出 x 的形状是 (batch_size, T, hidden_size)。
    """
    print("token IDs:", ids[0].tolist(), "input:", list(x.shape), flush=True)
    """
    ids[0].tolist()：取出第一个 batch 的 token ID，转换为 Python 的原生 list 进行打印（方便阅读）。
    list(x.shape)：打印词嵌入后张量的形状，通常形如 [1, 8, 4096]。
    """

    # A. 保存实际 forward 中的关键结果。
    """
    .detach()：截断梯度流，将其从 PyTorch 的动态计算图中分离出来，避免影响后续的反向传播，防止内存泄漏。
    .clone()：在显存/内存中开辟一段全新的空间复制该数据，防止后续原地操作（In-place operations）篡改该数值。
    handles 列表：存储所有注册的 Hook 句柄。PyTorch 在注册 hook 后会返回一个 handle，后续可以通过 handle.remove() 将这些 hook 注销，恢复模型原本状态。
    """
    trace = {"x": x.detach().clone()}
    handles = []

    """
    save_output：用于 register_forward_hook。当模块执行完毕后触发，获取其返回值 output 并存入 trace[name]。
    save_input：用于 register_forward_pre_hook。在模块执行之前触发，获取传入给该模块的输入实参 inputs[0] 并存入 trace[name]。
    """

    def save_output(name):
        def hook(module, inputs, output):
            trace[name] = output.detach().clone()

        return hook

    def save_input(name):
        def hook(module, inputs):
            trace[name] = inputs[0].detach().clone()

        return hook

    outputs = {
        "n": block.input_layernorm,
        "q": block.self_attn.q_proj,
        "k": block.self_attn.k_proj,
        "v": block.self_attn.v_proj,
        "attn_out": block.self_attn.o_proj,
        "z": block.post_attention_layernorm,
        "gate": block.mlp.gate_proj,
        "up": block.mlp.up_proj,
        "down": block.mlp.down_proj,
    }

    for name, module in outputs.items():
        handles.append(module.register_forward_hook(save_output(name)))

    for name, module in { 
        "a": block.self_attn.o_proj,               # 就是多头自注意力计算完成，传入 o_proj 前的隐层状态。
        "h": block.post_attention_layernorm,   # 就是第一处残差连接相加后的结果
        "act": block.mlp.down_proj,          # 在 SwiGLU 结构中，它是进入下投影层之前的激活值 
    }.items():
        handles.append(module.register_forward_pre_hook(save_input(name)))

    """
    当调用时如果指定了 output_attentions=True，Transformer 的 self_attn 模块会返回一个元组：
    output[0]：经过自注意力后的隐层表征向量。
    output[1]：注意力概率矩阵（经过 Softmax 后的 Attention Weights / Probabilities）。
    这里将注意力矩阵保存到了 trace["probs"] 中。
    """
    def save_attention(module, inputs, output):
        trace["probs"] = output[1].detach().clone()

    handles.append(block.self_attn.register_forward_hook(save_attention))

    original_rope = qwen2.apply_rotary_pos_emb

    def capture_rope(*rope_args, **rope_kwargs):
        q_rot, k_rot = original_rope(*rope_args, **rope_kwargs)
        trace["q_rot"] = q_rot.detach().clone()
        trace["k_rot"] = k_rot.detach().clone()
        return q_rot, k_rot

    qwen2.apply_rotary_pos_emb = capture_rope
    try:
        # 直接调用单层时必须自己传因果 mask：未来位置为 -inf。
        mask = torch.full((T, T), float("-inf")).triu(1)[None, None]
        y_full = block(
            x,
            attention_mask=mask,
            position_ids=pos,     # 每个 token 对应的位置索引。
            position_embeddings=model.model.rotary_emb(x, pos),    # Qwen2 新版结构要求传入计算好的旋转位置编码 cos/sin 缓存。
            use_cache=False,  # 关闭 KV Cache 缓存（当前做的是全序列计算验证，不是逐 Token 解码）。
        )
    finally:
        qwen2.apply_rotary_pos_emb = original_rope
        for handle in handles:
            handle.remove()

    trace["y"] = y_full.detach().clone()
    probs = trace["probs"]
    torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs.sum(-1)))
    assert torch.count_nonzero(probs.triu(1)).item() == 0

    # B. 同一层、同一输入，逐 token 执行。缓存只创建一次。
    """
    DynamicCache()：Hugging Face Transformers 库较新版本提供的动态缓存管理对象。它会随序列长度增加动态开辟空间，用于持久化保存每一步的 Key 和 Value 张量，避免显存反复重新分配。
    """
    cache = DynamicCache() 
    rows = []
    for t in range(T):
        # 输出 x 的形状是 (batch_size, T, hidden_size)。
        # 对输入序列切片，每次只取当前第 t 个位置的一个 Token（长度为 1）。
        #  pos: (1, T)
        xt, pt = x[:, t : t + 1], pos[:, t : t + 1]
        yt = block(
            xt,
            attention_mask=None,
            position_ids=pt,
            position_embeddings=model.model.rotary_emb(xt, pt),
            past_key_values=cache,
            use_cache=True,
            cache_position=pt[0],
        )
        rows.append(yt)
        assert cache.get_seq_length(0) == t + 1

    y_cached = torch.cat(rows, dim=1)    # 将 T 次循环得到的单 Token 向量沿序列维度（dim 1）拼接，恢复成 (1, T, hidden_size) 的完整矩阵。
    atol, rtol = 1e-5, 1e-4
    torch.testing.assert_close(y_cached, y_full, atol=atol, rtol=rtol)

    key_cache, value_cache = cache[0]  # [batch_size, num_key_value_heads, seq_len, head_dim]
    torch.testing.assert_close(key_cache, trace["k_rot"], atol=atol, rtol=rtol)

    v_heads = trace["v"].reshape(1, T, 2, 64).transpose(1, 2)
    torch.testing.assert_close(value_cache, v_heads, atol=atol, rtol=rtol)

    max_error = (y_cached - y_full).abs().max().item()
    print("完整因果 vs 缓存：PASS；最大绝对误差：", max_error, flush=True)

    # C. 写出原始形状、原始权重布局；不修改计算以适应 TianLLama。
    args.out.mkdir(parents=True, exist_ok=False)
    tensors = {}


    """
    dtype="<f4"：默认采用小端序（Little-Endian）的 32 位浮点数（IEEE 754 float32），兼容几乎所有现代 x86/ARM CPU 和 GPU。
    np.ascontiguousarray：保证数据在内存中是完全连续的（C-contiguous），防止多维切片带来的内存跨步（Stride）导致二进制数据错乱。
    assert np.isfinite(array).all()：严格的数值安全检查。只要数据中包含 NaN（非数值）或 Inf（无穷大），立刻报错阻断，防止导出有缺陷的坏数据。
    .tobytes(order="C")：去除所有结构头信息，只提取纯净的内存原生字节流。
    hashlib.sha256：为每个二进制文件计算 SHA-256 哈希值，用于后续在 C++/Rust 读取时进行完整性与防篡改校验。
    """
    def dump(name, tensor, dtype="<f4"):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=dtype)
        assert np.isfinite(array).all(), name
        payload = array.tobytes(order="C")
        filename = name + ".bin"
        (args.out / filename).write_bytes(payload)
        tensors[name] = {
            "file": filename,
            "shape": list(array.shape),
            "dtype": dtype,
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    modules = {
        "attention_norm": block.input_layernorm,
        "ffn_norm": block.post_attention_layernorm,
        "wq": block.self_attn.q_proj,
        "wk": block.self_attn.k_proj,
        "wv": block.self_attn.v_proj,
        "wo": block.self_attn.o_proj,
        "gate": block.mlp.gate_proj,
        "up": block.mlp.up_proj,
        "down": block.mlp.down_proj,
    }
    for name, module in modules.items():
        dump("w." + name, module.weight)
    for name, module in {
        "bq": block.self_attn.q_proj,
        "bk": block.self_attn.k_proj,
        "bv": block.self_attn.v_proj,
    }.items():
        dump("w." + name, module.bias)

    dump("token_ids", ids[0], "<i4")
    dump("positions", pos[0], "<i4")
    for name, tensor in trace.items():
        if name in ("q_rot", "k_rot"):
            tensor = tensor[0].transpose(0, 1)  # [H,T,D] -> [T,H,D]
        else:
            tensor = tensor[0]  #  去除 Batch 维度：通过 tensor[0] 剥离掉单 batch 维度，简化后端测试实现的复杂度。
        dump("ref." + name, tensor)  # 所有前文 hook 拦截的中间量（如注意力权重、MLP 中间激活等）以及缓存对比值都存为 ref.*.bin，方便第三方引擎逐个算子进行 Diff 比对。
    dump("ref.y_cached", y_cached[0])
    dump("ref.k_cache", key_cache[0].transpose(0, 1))
    dump("ref.v_cache", value_cache[0].transpose(0, 1))

    manifest = {
        "format_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "layer_index": 0,
        "device": "cpu",
        "compute_dtype": "float32",
        "attention_backend": "eager",
        "weight_layout": "out_in",  # 明确指示权重矩阵是 [out_features, in_features] 形状。
        "rope_layout": "half_split",   # 明确指示旋转位置编码是前半/后半切分（RoPE 有两种主流切法：相邻切分 vs 前后半切分）。
        "python": platform.python_version(),
        "packages": {
            p: metadata.version(p)
            for p in (
                "torch",
                "transformers",
                "numpy",
                "huggingface_hub",
                "safetensors",
            )
        },
        "prompt": prompt,
        "chat_template": False,
        "add_special_tokens": False,
        "max_input_tokens": 8,
        "token_ids": ids[0].tolist(),
        "config": config.to_dict(),
        "validation": {
            "passed": True,
            "atol": atol,
            "rtol": rtol,
            "cached_vs_full_max_abs": max_error,
        },
        "tensors": tensors,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("导出完成：", args.out.resolve(), "；Tensor 数量：", len(tensors))


if __name__ == "__main__":
    main()
