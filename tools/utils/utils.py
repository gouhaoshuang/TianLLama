import argparse
from pathlib import Path
import transformers
import hashlib

import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def load_model(MODEL_ID, REVISION):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            dtype=torch.float32,
            attn_implementation="eager",
            use_safetensors=True,
            trust_remote_code=False,
        )
        .cpu()
        .eval()
    )
    return tokenizer, model

def as_f32_array(value):
    # 先转 FP32，再统一为小端、C 连续布局；不转置矩阵。
    array = np.ascontiguousarray(
        value.detach().cpu().float().numpy(), dtype="<f4"
    )
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Tensor must be non-empty and finite")
    return array


"""
data = memoryview(array).cast("B"):
    零拷贝( Zero-Copy )视图。将数组以“单字节( unsigned char )”的形式透视出来，
    绝对不拷贝数据，也不申请额外的内存。
digest = hashlib.sha256():
    hashlib: Python 内置的加密哈希库，提供了 MD5、SHA-1、SHA-256 等多种安全哈希算法。
    sha256(): 一种被广泛使用的密码学哈希算法( Secure Hash Algorithm 256-bit )。它的特点是: 
        输出固定: 无论输入多大的数据( 几个字节还是几百 GB )，最终都会输出一个 256 位( 32 字节，转成 16 进制字符串就是 64 个字符 ) 的哈希值。
        单向不可逆: 无法通过哈希值反推原始数据。
        雪崩效应: 原始数据哪怕只改动了 1 个 bit，生成的哈希值也会发生翻天覆地的变化。
    digest: 保存这个哈希对象的变量名( “digest” 本身就是密码学中的术语，意为“信息摘要” )。
    # 一次性算出哈希( 适合小数据 )
    hash_val = hashlib.sha256(data).hexdigest()  # 吐出最终的 64 位十六进制字符串
    
    初始化: digest = hashlib.sha256() ( 创建一个空的计算状态机 )
    增量喂数据: digest.update(chunk) ( 每次只喂 1 MiB 数据，在循环中反复调用 )
    最终收网: digest.hexdigest() ( 所有数据喂完后，吐出最终的 64 位十六进制字符串 )
1 << 20
    位运算:= 1 MiB。这是极其理想的 I/O 缓冲区大小。

    
""" 
def write_array(file, array):
    # 分块写入并计算该 Tensor 的 SHA256，避免为大 Embedding 再生成一整份 bytes。
    data = memoryview(array).cast("B")

    digest = hashlib.sha256()
    for start in range(0, len(data), 1 << 20):
        chunk = data[start:start + (1 << 20)]
        if file.write(chunk) != len(chunk):
            raise OSError("Short write while exporting tensor")
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


"""
dump_weight:
    file: 一个以追加/写入二进制模式( 如 "wb" )打开的大文件对象。所有的权重都会被紧密拼接到这一个文件里。
    index: 一个字典，用来收集元数据( 最终会保存为类似 model.json 的索引清单 )。
    name: 当前权重的名称，如 "model.embed_tokens.weight"。
    value: PyTorch 的张量( Tensor )对象。

file.tell(): 
    获取当前文件指针的位置（即距离文件开头的字节数）。这个值就是当前张量在二进制大文件里的起始地址。
"""
def dump_weight(file, index, name, value):
    if name in index:
        raise ValueError("Duplicate weight: " + name)
    
    array = as_f32_array(value)

    offset = file.tell()
    if offset % 4 != 0:
        raise ValueError("Unaligned FP32 weight offset")
    
    digest = write_array(file, array)

    if file.tell() - offset != array.nbytes:
        raise OSError("Unexpected byte count: " + name)
    
    index[name] = dict(
        offset_bytes=offset, shape=list(array.shape), dtype="<f4",
        nbytes=int(array.nbytes), sha256=digest,
    )


def dump_reference(root, index, name, value):

    # name 仅由下方固定用例产生，不接受用户提供的路径。
    if name in index:
        raise ValueError("Duplicate reference: " + name)
    
    array = as_f32_array(value)

    path = root / (name + ".bin")

    with path.open("xb") as file:
        digest = write_array(file, array)

    if path.stat().st_size != array.nbytes:
        raise OSError("Unexpected reference byte count: " + name)
    
    index[name] = dict(
        file=path.name, shape=list(array.shape), dtype="<f4",
        nbytes=int(array.nbytes), sha256=digest,
    )
