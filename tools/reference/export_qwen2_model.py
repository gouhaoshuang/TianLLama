import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen2 import modeling_qwen2 as qwen2
from tools.utils.utils import (
    load_model,
    dump_weight,
    dump_reference,
    sha256_file,
)

"""
cd /data/ghs/TianLLama
/data/ghs/miniconda3/envs/agent/bin/python -m tools.reference.export_qwen2_model \
  --out /data/ghs/TianLLama/test_data/qwen2_5_0_5b_model_v2
"""

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

DEFAULT_SNAPSHOT = (
    Path("/data/ghs/.cache/huggingface/hub")
    / "models--Qwen--Qwen2.5-0.5B-Instruct"
    / "snapshots"
    / REVISION
)
TOKENIZER_AND_CONFIG_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        required=False,
        help="本地模型 snapshot 目录，默认：%(default)s",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError("不覆盖已有目录，请使用新输出路径")
    if transformers.__version__ != "4.57.6":
        raise RuntimeError("请使用第 11 课的 transformers==4.57.6 环境")
    for name in TOKENIZER_AND_CONFIG_FILES:
        if not (args.snapshot / name).is_file():
            raise FileNotFoundError(f"snapshot 缺少文件：{args.snapshot / name}")

    torch.set_num_threads(1)
    torch.manual_seed(0)

    print("加载", MODEL_ID, REVISION, flush=True)
    print("本地 snapshot: ", args.snapshot, flush=True)
    tokenizer, model = load_model(MODEL_ID, REVISION)
    config = model.config

    assert (
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.vocab_size,
    ) == (896, 4864, 24, 14, 2, 151936)

    assert config.tie_word_embeddings and not config.use_sliding_window
    assert config.rms_norm_eps == 1e-6 and config.rope_theta == 1000000.0
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert torch.equal(
        model.get_input_embeddings().weight, model.get_output_embeddings().weight
    )

    args.out.mkdir(parents=True, exist_ok=False)
    for name in TOKENIZER_AND_CONFIG_FILES:
        shutil.copyfile(args.snapshot / name, args.out / name)

    # A. 所有正式权重只写入这一个文件。
    tensors = {}
    weights_path = args.out / "weights.bin"
    with weights_path.open("xb") as weights_file:
        dump_weight(
            weights_file, tensors, "embedding", model.get_input_embeddings().weight
        )
        dump_weight(weights_file, tensors, "final_norm", model.model.norm.weight)
        # lm_head 与 embedding 共享，不写第二份。
        for i, block in enumerate(model.model.layers):
            print(f"导出层 {i + 1}/{config.num_hidden_layers}", flush=True)
            prefix = f"layers.{i}."
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
                dump_weight(weights_file, tensors, prefix + name, module.weight)
            for name, module in (
                ("bq", block.self_attn.q_proj),
                ("bk", block.self_attn.k_proj),
                ("bv", block.self_attn.v_proj),
            ):
                dump_weight(weights_file, tensors, prefix + name, module.bias)

    # 2 个模型级权重 + 每层 9 个 weight 和 3 个 bias，共 290 项。
    if len(tensors) != 2 + 12 * config.num_hidden_layers:
        raise ValueError("Unexpected weight count")
    total_bytes = weights_path.stat().st_size
    cursor = 0
    for entry in tensors.values():
        if entry["offset_bytes"] != cursor:
            raise ValueError("Weight index contains a gap or overlap")
        cursor += entry["nbytes"]
    if cursor != total_bytes:
        raise ValueError("Weight index does not cover weights.bin")

    weights_info = dict(
        file=weights_path.name,
        nbytes=total_bytes,
        sha256=sha256_file(weights_path),
    )

    # B. Tokenizer 的独立参考用例；decode 不假定一定等于原文。
    # 例如 NFC 会把 e + combining acute 转成预组合字符。
    texts = [
        "",
        "你好，请介绍一下你自己。",
        "Hello, world!",
        " hello  world ",
        "a\nb\t中文",
        "1234567890",
        "🙂🚀",
        "é",
        "e\u0301",
        "Ġ",
        "<|im_end|>",
        "<tool_call>{}</tool_call>",
    ]

    cases = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        cases.append(
            dict(
                text=text,
                ids=ids,
                decoded=tokenizer.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                ),
                decoded_skip=tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                ),
            )
        )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "你好"},
    ]
    """
    tokenize=False（赋值给 chat_text）：
        只做模板渲染，不切词，输出一个完整的字符串。
    对于 Qwen2.5，渲染出的 chat_text 字符串长这样：
        <|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant
    """
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chat_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    (args.out / "tokenizer_cases.json").write_text(
        json.dumps(
            dict(
                model_id=MODEL_ID,
                revision=REVISION,
                cases=cases,
                chat_text=chat_text,
                chat_ids=chat_ids,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # C. 测试参考答案单独保存，不追加到 weights.bin。
    short_ids = tokenizer.encode("你好，请介绍一下你自己。",
                                 add_special_tokens=False)[:5]
    references = {}
    reference_tensors = {}
    for name, token_ids in (("short", short_ids), ("chat", chat_ids)):

        if not token_ids:
            raise ValueError("Empty reference input: " + name)
        print("计算参考答案：", name, flush=True)

        ids = torch.tensor([token_ids], dtype=torch.long)

        full = model(input_ids=ids, use_cache=False).logits[0]

        cache = None
        rows = []

        for t in range(ids.shape[1]):

            out = model(input_ids=ids[:, t:t+1], past_key_values=cache,
                        use_cache=True)
            cache = out.past_key_values
            rows.append(out.logits[0, 0])

        cached = torch.stack(rows)

        # 这里只比较两个 PyTorch 执行路径，不是 C++ 已通过。
        torch.testing.assert_close(full, cached, atol=2e-4, rtol=2e-4)

        dump_reference(args.out, reference_tensors, "ref." + name + ".logits", full)
        dump_reference(args.out, reference_tensors, "ref." + name + ".cached_logits", cached)
        dump_reference(args.out, reference_tensors, "ref." + name + ".embedding",
                       model.get_input_embeddings()(ids)[0])
        
        references[name] = dict(
            token_ids=token_ids,
            cached_vs_full_max_abs=float((full-cached).abs().max()))
        del cache, out, rows, full, cached

    manifest = dict(
        format_version=2, model_id=MODEL_ID, revision=REVISION,
        compute_dtype="float32", weight_layout="out_in", rope_layout="half_split",
        transformers=transformers.__version__, torch=torch.__version__,
        config=config.to_dict(), lm_head_alias="embedding",
        weights_file=weights_info, tensors=tensors,
        references=references, reference_tensors=reference_tensors)
    # 最后发布索引；参考计算或写文件失败时不生成最终 manifest。
    # 同目录重命名避免正常读取者看见半份 JSON，不保证掉电持久性。
    manifest_tmp = args.out / "manifest.json.tmp"
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_tmp.replace(args.out / "manifest.json")
    print("导出完成", args.out)
    print("权重数：", len(tensors), "权重总字节数：", total_bytes)



if __name__ == "__main__":
    main()
