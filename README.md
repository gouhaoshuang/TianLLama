# TianLLama

从零逐步实现的 C++ / CUDA 大语言模型推理框架，参考 KuiperLLama 的设计与课程，围绕真实模型学习内存管理、张量、算子、KV Cache 和自回归推理。

项目以 **Qwen2.5-0.5B-Instruct** 为首个目标模型。目前已实现 CPU FP32 全模型前向、Tokenizer 和贪心生成 Demo，同时保留基础算子的 CUDA 实现。当前重点是理解实现、验证数值正确性，再逐步完善 GPU 执行与性能。

> 这是持续迭代中的学习型项目，不是生产级推理服务。基础算子支持 CUDA，不代表完整模型已经支持 GPU 推理。

## 当前能力

| 模块       | 已实现内容                                                                            |
| ---------- | ------------------------------------------------------------------------------------- |
| 内存与设备 | CPU / GPU 分配器、内存复制与清零、Buffer RAII                                         |
| Tensor     | 连续存储、共享底层 Buffer 的视图、偏移与重叠检查                                      |
| 基础算子   | Add、RMSNorm、带可选 bias 的 Linear、SwiGLU、RoPE、稳定 Softmax；包含 CPU / CUDA 实现 |
| Attention  | CPU MHA / GQA / MQA，固定容量、逐 token 更新的 KV Cache                               |
| 模型组件   | 通用 CPU DecoderBlock，以及适配 Qwen2 的 Attention / DecoderLayer                     |
| 完整模型   | 权重加载、Embedding 查表、24 层前向、最终 RMSNorm、共享权重的词表投影                 |
| Tokenizer  | 基于 tokenizers-cpp 的编码与解码、单轮 Qwen 聊天模板                                  |
| 生成 Demo  | CPU FP32 贪心生成、结束 token 与长度限制、独立请求重置                                |
| 正确性验证 | 小数据单元测试、真实单层参考、全模型 logits、Tokenizer 与聊天输入对齐测试             |

当前推理计算以 FP32 为主。数据类型枚举中存在 FP16 / Int8，不表示已完成相应推理或量化支持。

## 推理流程

```text
用户文本 → 聊天模板 → Tokenizer → token IDs
                                      ↓
                                 Embedding 查表
                                      ↓
                       Qwen2DecoderLayer × 24 + KV Cache
                                      ↓
                           最终 RMSNorm → 输出 Linear
                                      ↓
                                  logits → argmax
                                      ↓
                        新 token 回送模型，解码并输出文本
```

Python 仅用于模型资产准备、权重导出和生成参考答案；C++ Demo 的推理过程不启动 Python。

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
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j 4
```

`89` 是项目当前默认的 CUDA 架构值；使用其他 GPU 时应按目标硬件调整。学习调试时可以使用独立的 Debug 构建目录，性能测量应使用 Release。

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
ctest --test-dir build --output-on-failure \
  -E 'Qwen2Layer0Test|QwenTokenizerTest|QwenModelRealTest'
```

准备好两套参考数据后，运行当前注册到 CTest 的测试：

```bash
ctest --test-dir build --output-on-failure
```

**当前 CMake 中全模型测试的自动注册被注释，但仍会构建 `test_qwen_model`。** 全模型测试需单独运行：

```bash
./build/test_qwen_model --gtest_filter=QwenModelRealTest.ShortLogitsAndReset
./build/test_qwen_model --gtest_filter=QwenModelRealTest.ChatTokenizerAndLogits
```

前者比较短序列每个位置的完整 logits，并检查 reset 后重新推理；后者连接聊天模板、C++ Tokenizer 和完整模型，与聊天参考 logits 比较。CPU 全模型测试比算子测试耗时更长。

测试数据目录由 `CMakeLists.txt` 中的编译定义指定。缺少数据时，相关测试会失败，不能把未运行真实模型测试视为全模型验证通过。

## 运行聊天 Demo

```bash
./build/qwen_chat
```

当前 Demo 的配置写在 `demo/qwen_chat.cpp` 中：

| 配置              | 当前值                                                  |
| ----------------- | ------------------------------------------------------- |
| 模型目录          | `/data/ghs/TianLLama/test_data/qwen2_5_0_5b_model_v2` |
| KV Cache 容量     | 1024 token                                              |
| 最大新增 token 数 | 512                                                     |
| 选择方式          | Greedy / argmax                                         |
| 退出指令          | `/exit`                                               |

当前不解析命令行模型路径参数。在其他机器或目录使用时，需要手动修改上述源码配置并重新编译。每次提问会 reset，**不保留上一轮对话历史**；prompt 与生成预算之和不能超过缓存容量。

Demo 的文本输出仍在调试：当前逐 token 单独解码，中文或跨 token 的 UTF-8 字节可能显示为替换字符。代码中已有累积解码辅助函数，但当前输出路径未启用它，不应把逐 token 显示当作完善的流式解码实现。

## 后续方向

- 完善生成结果对齐、UTF-8 流式输出、异常处理和命令行配置。
- 补齐 GPU KV Cache、GQA Attention 与 Qwen2 全模型执行路径。
- 在 CPU / GPU 数值对齐后建立性能基准，再优化工作区复用与内核同步。
- 根据实际需要逐步探索量化、批量推理和更多模型，不提前增加复杂抽象。

## 参考与致谢

- KuiperLLama：框架设计、算子实现与配套课程参考。
- Qwen：目标模型及模型资产。
- Hugging Face Transformers：Python 参考推理与 Tokenizer 数据。
- tokenizers-cpp、nlohmann/json、GoogleTest：分词绑定、配置解析与测试基础设施。

模型资产及第三方组件遵循各自的许可与使用条款。
