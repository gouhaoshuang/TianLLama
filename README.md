# TianLLama

从零逐步实现的 C++ / CUDA 大语言模型推理框架，参考 KuiperLLama 的设计与课程，围绕真实模型学习内存管理、张量、算子、KV Cache 和自回归推理。

项目以 **Qwen2.5-0.5B-Instruct** 为首个目标模型。目前已实现 CPU / GPU FP32 全模型前向、Tokenizer、GPU 贪心生成 Demo，以及 prompt / 生成耗时和 token 输出速率记录。当前重点是验证完整模型的数值正确性，逐步减少 GPU 执行中的数据回读、临时分配和同步开销。

> 当前是单 GPU、单请求、固定容量 KV Cache 的学习型实现。GPU 路径使用默认 stream 和同步执行，部分有限值检查会将数据读回 CPU，性能优化仍在推进。

## 当前能力

模块已实现内容内存与设备CPU / GPU 分配器、内存复制与清零、Buffer RAIITensor连续存储、共享底层 Buffer 的视图、偏移与重叠检查基础算子Add、RMSNorm、带可选 bias 的 Linear、SwiGLU、RoPE、稳定 Softmax；包含 CPU / CUDA 实现AttentionCPU MHA / GQA / MQA、CUDA GQA Attention，CPU / GPU 固定容量 KV Cache模型组件通用 CPU DecoderBlock，以及支持 CPU / GPU 的 Qwen2 Attention / DecoderLayer完整模型CPU / GPU 设备选择、逐份权重上传、Embedding 行复制、24 层前向、最终 RMSNorm 和共享词表投影Tokenizer基于 tokenizers-cpp 的编码与解码、单轮 Qwen 聊天模板生成 DemoGPU FP32 贪心生成、GPU logits 读回后 CPU argmax、累积解码、结束条件与独立请求重置性能记录prompt token 数与前向耗时、实际生成 token 数、生成耗时和输出 tokens/s正确性测试算子与 Cache 小数据测试、真实 CPU 单层参考、Tokenizer 对齐、GPU 全模型 logits 与 reset

当前推理计算以 FP32 为主。数据类型枚举中存在 FP16 / Int8，不表示已完成相应推理或量化支持。

## GPU Demo 推理流程

```text
用户文本 → 聊天模板 → Tokenizer → token IDs
                                      ↓
                              GPU Embedding 行复制
                                      ↓
                   GPU Qwen2DecoderLayer × 24 + GPU KV Cache
                                      ↓
                         GPU 最终 RMSNorm → 输出 Linear
                                      ↓
                           GPU logits → 读回 CPU → argmax
                                      ↓
                        新 token 回送模型，解码并输出文本
```

Python 仅用于模型资产准备、权重导出和生成参考答案；C++ Demo 的推理过程不启动 Python。

模型接口默认选择 CPU，也可显式选择 GPU：

```cpp
auto cpu_model = model::Qwen2Model::load(root, capacity);
auto gpu_model = model::Qwen2Model::load(root, capacity, base::DeviceType::kDeviceGPU);
```

GPU 模式下，权重、中间 Tensor、每层 KV Cache 和词表投影均位于 GPU；CPU 负责分词、调度、最终 token 选择和文本输出。Embedding 与输出投影共享同一份权重，GPU 加载不重复上传词表矩阵。

## 目录结构

```text
TianLLama/
├── include/             # 基础类型、Tensor、算子、模型及 Tokenizer 接口
├── source/
│   ├── base/            # 内存、缓存、状态与二进制读取
│   ├── tensor/          # Tensor 实现
│   ├── op/              # 算子参数校验与后端分发
│   ├── kernel/          # CPU / CUDA 计算内核
│   ├── model/           # DecoderBlock 与 Qwen2 模型
│   └── tokenizer/       # Qwen Tokenizer 与聊天模板
├── demo/                # qwen_chat 交互示例
├── test/                # 单元测试与真实模型对齐测试
├── tools/               # Python 权重导出与参考数据工具
├── test_data/           # 本地模型和参考数据，不随仓库提交
└── docs/                # 本地课程记录与逐步实现笔记
```

`docs/` 和 `test_data/` 当前被 `.gitignore` 忽略；新克隆仓库不保证包含这些目录中的资料。

## 构建

### 环境要求

- 支持 C++20 的 C++ 编译器，以及 C 编译器。
- CMake 3.19 或更高版本、Git。
- CUDA Toolkit / nvcc；CUDA 源文件使用 C++17。
- GoogleTest，需要能被 `find_package(GTest CONFIG REQUIRED)` 找到。
- Rust / Cargo，用于构建 Tokenizer 的 Rust 绑定。
- 首次构建依赖时需要网络访问，或已准备好依赖缓存。

CMake 通过 FetchContent 获取 nlohmann/json 和固定提交的 tokenizers-cpp。**当前没有关闭 CUDA 的构建选项，即使只运行 CPU 模型，也需要 CUDA 编译环境。**

在项目根目录执行：

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-release -j 4
```

`89` 是项目当前默认的 CUDA 架构值；使用其他 GPU 时应按目标硬件调整。学习调试时可以使用独立的 Debug 构建目录，性能测量应使用 Release。运行 GPU Demo 和 CUDA 测试还需要可用的 NVIDIA GPU 与驱动。

## 模型与参考数据

当前导出工具固定使用：

```text
model_id : Qwen/Qwen2.5-0.5B-Instruct
revision : 7ae557604adf67be50417f59c2c2f167def9a775
计算精度 : FP32
权重布局 : [out_features, in_features]
RoPE布局 : half_split
```

模型配置为隐藏维度 896、FFN 维度 4864、24 层、14 个 Q 头 / 2 个 KV 头、每头维度 64。Embedding 与输出投影共享权重。

### 准备资产

1. 准备上述固定 revision 的 Hugging Face 模型缓存，包括权重、Tokenizer 和配置文件。
2. 准备 Python 环境：PyTorch、NumPy、Hugging Face Hub、Safetensors，以及 **`transformers==4.57.6`**。导出脚本会检查 Transformers 版本。
3. 从项目根目录运行下列导出命令。

当前 `tools/utils/utils.py` 按固定模型 ID / revision 加载模型，其中 Tokenizer 使用 `local_files_only=True`，因此不能只准备一个空输出目录就直接导出。

### 导出第 0 层参考数据

```bash
python -m tools.reference.export_qwen2_layer0 \
  --out test_data/qwen2_5_0_5b_layer0_v1
```

### 导出完整模型与参考数据

将下面的 snapshot 路径替换为本机同一 revision 的缓存目录：

```bash
python -m tools.reference.export_qwen2_model \
  --snapshot /path/to/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
  --out test_data/qwen2_5_0_5b_model_v2
```

注意：当前 `--snapshot` 用于检查和复制配置、Tokenizer 文件，**不替代模型 ID / revision 的加载来源**。该目录必须与模型缓存保持一致，不能混用 Base / Instruct 或其他版本资产。

完整导出目录主要包含：

```text
qwen2_5_0_5b_model_v2/
├── manifest.json           # 配置、权重偏移、形状及参考数据索引
├── weights.bin             # FP32 权重；共享权重不重复写入
├── tokenizer.json
├── tokenizer_config.json
├── config.json
├── generation_config.json
├── tokenizer_cases.json    # 分词与聊天模板参考
└── ref.*.bin               # short / chat 等参考张量
```

两个导出脚本都拒绝覆盖已有输出目录。若数据已完整生成，可直接复用；重新导出时使用新目录，并同步调整调用处的数据路径。约 0.5B 参数的 FP32 权重需要约 2 GB 存储，导出与加载还需要额外内存和磁盘空间。

## 运行测试

无需真实模型数据的测试：

```bash
ctest --test-dir build-release --output-on-failure \
  -E 'Qwen2Layer0Test|QwenTokenizerTest|QwenModelRealTest'
```

这组测试仍包含 CUDA 算子和 GPU Cache 测试，需要可用的 GPU。

准备好两套参考数据后，运行当前注册到 CTest 的测试：

```bash
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build-release --output-on-failure
```

GPU Cache 与 Attention 的小数据测试也可以单独运行：

```bash
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build-release --output-on-failure \
  -R '^(KVCacheTest|AttentionCudaTest)\.'
```

**全模型测试已注册到 CTest。** 当前启用的用例是 `QwenModelRealTest.GpuShortLogitsAndReset`，也可直接执行：

```bash
CUDA_VISIBLE_DEVICES=0 ./build-release/test_qwen_model \
  --gtest_filter=QwenModelRealTest.GpuShortLogitsAndReset
```

该测试运行真实模型的全部 24 层，检查短序列每个位置的完整 GPU logits、各层 Cache 长度，以及 reset 后首 token 的结果。GPU 输出读回 CPU 后与参考 Tensor 比较，当前误差阈值为 `1e-3 + 1e-4 * abs(reference)`。

原来的 CPU 全模型 `ShortLogitsAndReset` 和 `ChatTokenizerAndLogits` 用例目前在源码中被注释，不参与测试；CPU 真实单层和独立 Tokenizer 测试仍保留。

测试数据目录由 `CMakeLists.txt` 中的编译定义指定。缺少数据时，相关测试会失败，不能把未运行真实模型测试视为全模型验证通过。

## 运行聊天 Demo

```bash
CUDA_VISIBLE_DEVICES=0 ./build-release/qwen_chat \
  test_data/qwen2_5_0_5b_model_v2
```

第一个命令行参数是模型导出目录。省略参数时，程序使用源码中的默认目录：

| 配置              | 当前值                                                  |
| ----------------- | ------------------------------------------------------- |
| 模型目录          | `/data/ghs/TianLLama/test_data/qwen2_5_0_5b_model_v2` |
| 执行设备          | GPU / FP32                                              |
| KV Cache 容量     | 1024 token                                              |
| 最大新增 token 数 | 512                                                     |
| 选择方式          | Greedy / argmax                                         |
| 退出指令          | `/exit`                                               |

KV Cache 容量、生成预算和执行设备当前仍写在 `demo/qwen_chat.cpp` 中，没有对应的命令行选项。每次提问会 reset，**不保留上一轮对话历史**；prompt 与最大生成预算之和不能超过缓存容量，超过时会报错。

文本输出使用累积 token IDs 解码，只打印新增的完整 UTF-8 字符。生成遇到结束 token 或达到长度上限后停止；当前 EOS 为 `<|im_end|>` 和 `<|endoftext|>`。模型与 Tokenizer 只在启动时加载一次。

### 性能记录

每次回答结束后，Demo 输出一行 `[性能]` 记录：

| 字段                           | 含义                                                 |
| ------------------------------ | ---------------------------------------------------- |
| `prompt` / `prefill`       | 输入 token 数与逐 token prompt 前向耗时，单位 ms     |
| `generated` / `generation` | 实际生成 token 数（不含 EOS）与生成阶段耗时，单位 ms |
| `output`                     | 实际生成 token 数除以生成阶段秒数，单位 tokens/s     |

统计不包含模型加载、聊天模板构造和分词。生成耗时包含模型前向、GPU logits 读回、CPU argmax、解码和终端打印，因此 `output` 是当前 Demo 的输出速率，不是纯 GPU kernel 吞吐，也不是包含 prompt 耗时的总请求吞吐。

第一枚生成 token 使用最后一次 prompt 前向的 logits；达到生成上限时，输出 N 枚 token 通常只需 N−1 次生成阶段前向。只生成零个或一个 token 时，不适合据此判断持续生成速度。

使用 Release 构建，在同一进程先运行一个短请求预热，再重复相同问题记录结果。对比时保持 GPU、prompt、缓存容量、生成预算和打印方式一致，并记录实际生成数量。

## 后续方向

- 恢复 CPU 全模型回归，扩展 GPU 聊天输入与生成结果对齐。
- 减少模型与 Attention 的有限值回读，避免加载时重复扫描 GPU 权重。
- 复用 Attention 的 scores / probs 工作区，再逐步减少逐算子同步。
- 跳过非最后一个 prompt token 的词表投影，进一步优化 prefill。
- 完善请求级异常处理与容量、生成长度等命令行配置。
- 根据实际需要逐步探索量化、批量推理和更多模型，不提前增加复杂抽象。

## 参考与致谢

- KuiperLLama：框架设计、算子实现与配套课程参考。
- Qwen：目标模型及模型资产。
- Hugging Face Transformers：Python 参考推理与 Tokenizer 数据。
- tokenizers-cpp、nlohmann/json、GoogleTest：分词绑定、配置解析与测试基础设施。

模型资产及第三方组件遵循各自的许可与使用条款。
