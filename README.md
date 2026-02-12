# FedLoRA — 基于 Ray 的联邦 LoRA 微调框架

> 在多机多卡 GPU 集群上，通过 **Ray** 调度 + **LoRA** 参数高效微调 + **FedAvg 加权聚合**，对大语言模型（如 Qwen3-1.7B）进行联邦学习式的医学问答 SFT 训练。

---

## 目录

- [项目概览](#项目概览)
- [整体架构](#整体架构)
- [目录结构](#目录结构)
- [环境依赖](#环境依赖)
- [集群部署与 Ray 启动](#集群部署与-ray-启动)
  - [前置条件](#前置条件)
  - [启动 Ray Head 节点](#启动-ray-head-节点)
  - [启动 Ray Worker 节点](#启动-ray-worker-节点)
  - [验证集群状态](#验证集群状态)
- [数据准备](#数据准备)
- [模型准备](#模型准备)
- [配置说明](#配置说明)
- [运行联邦训练](#运行联邦训练)
- [单机训练（可选）](#单机训练可选)
- [关键模块详解](#关键模块详解)
- [常见问题](#常见问题)
- [硬件环境说明](#硬件环境说明)

---

## 项目概览

本项目实现了一个 **联邦学习 + LoRA 微调** 的训练管线，核心思路：

1. **多个 Client** 分布在不同机器/GPU 上，各自持有私有数据集（如不同语言的 MedQA）。
2. 每个 Client 在本地使用 HuggingFace `Trainer` 对基座模型进行 LoRA SFT 训练若干步。
3. **Aggregator** 收集所有 Client 的 LoRA 权重，按样本数进行 **加权平均**（FedAvg），生成全局 LoRA。
4. 全局 LoRA 下发给各 Client，进入下一轮。
5. 使用 **Ray** 实现跨机器的任务调度、GPU 资源分配和参数通信。

**典型应用场景**：多机构/多数据源的医学大模型协同微调，数据不出域。

---

## 整体架构

```
┌────────────────────────────────────────────────────────────────┐
│                      Ray Head Node (主节点)                     │
│                                                                │
│   main_new.py                                                  │
│   ┌──────────────┐    distribute     ┌──────────────────────┐  │
│   │  Aggregator   │ ──────────────►  │   Client 1 (GPU 0)   │  │
│   │  (Ray Actor)  │                  │   - 本地 LoRA 训练    │  │
│   │              │ ◄──────────────── │   - 回传 lora_state   │  │
│   │  FedAvg 聚合  │    upload         │   - 数据: MedQA_EN   │  │
│   │              │                   └──────────────────────┘  │
│   │              │    distribute     ┌──────────────────────┐  │
│   │              │ ──────────────►  │   Client 2 (GPU 1)   │  │
│   │              │                  │   - 本地 LoRA 训练    │  │
│   │              │ ◄──────────────── │   - 回传 lora_state   │  │
│   └──────────────┘    upload         │   - 数据: MedQA_CN   │  │
│                                      └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

每轮（Round）流程：

```
Aggregator.distribute_parameters()     # 下发全局 LoRA（首轮为空字典）
    ↓
Client.process_parameters(lora_dict)   # 加载全局 LoRA → 本地训练 N 步 → 回传本地 LoRA
    ↓
Aggregator.aggregate(client_results)   # 加权平均所有 Client 的 LoRA → 更新全局状态
    ↓
进入下一轮 ...
```

---

## 目录结构

```
FedLora/
├── main_new.py              # 🚀 联邦训练主入口（配置 + 启动 Ray + 训练循环）
├── client_new.py            # 🤖 联邦客户端 Ray Actor（加载模型、本地训练、回传 LoRA）
├── aggregator.py            # 📊 聚合器 Ray Actor（FedAvg 加权聚合 LoRA 权重）
├── train_fed_jt.py          # 🔧 训练核心（build_components / make_trainer_for_steps）
├── conversation_COT.py      # 💬 对话模板管理（Qwen3/LLaMA3 等多种 prompt 模板）
├── data/
│   └── MedQA_EN.jsonl       # 📄 英文医学问答数据集（10178 条）
├── utils/
│   ├── test_maca.py         # 🔍 沐曦硬件环境检测（历史遗留，NVIDIA 环境无需使用）
│   └── test_prob.py         # 🔍 集群节点 GPU 探测脚本
├── old/                     # 📁 早期版本代码（留档参考）
│   ├── client.py            # 旧版客户端（手动训练循环）
│   ├── main.py              # 旧版主入口
│   ├── train_utils.py       # 旧版训练工具（toy data 单步训练）
│   ├── train_lora_only_ans.py  # 单机 LoRA 微调脚本
│   └── readme.md
└── README.md                # 📖 本文件
```

---

## 环境依赖

### Python 版本

- Python >= 3.10

### 核心依赖

```txt
torch >= 2.0 (with CUDA support)
transformers >= 4.40
peft >= 0.10
datasets
ray >= 2.10
pandas
numpy
swanlab                 # 可选：训练日志可视化平台
modelscope              # 可选：模型下载（如使用 ModelScope 源）
```

### 安装

```bash
pip install torch transformers peft datasets ray pandas numpy swanlab
```

### 硬件要求

- **GPU**：每个 Client + Aggregator 各需 **1 张 GPU**（默认配置下共需 3 张 GPU）
- **显存**：Qwen3-1.7B + LoRA 约需 **8-10 GB** / 卡（bf16）
- **网络**：多机部署需要节点间网络互通（Ray 默认使用 6379 端口 + 随机端口）

> **注意**：本项目最初在沐曦（MetaX）MACA 硬件上开发，当前已迁移至 **NVIDIA GPU** 环境。代码中残留的 `MACA_PATH`、`/opt/maca` 等环境变量为历史遗留，在 NVIDIA 环境中 **无需设置**，可忽略或删除。

---

## 集群部署与 Ray 启动

### 前置条件

1. **所有节点**安装相同的 Python 环境和依赖包。
2. **所有节点**能通过网络互相访问（SSH 免密推荐但非必需）。
3. **所有节点**上模型文件路径一致（如 `/exp1/Qwen3/Qwen3-1.7B/`）。
4. **所有节点**上能访问各自的训练数据集（数据无需共享，联邦学习的数据留在本地）。
5. NVIDIA 驱动 & CUDA 正常工作（`nvidia-smi` 能看到 GPU）。

### 启动 Ray Head 节点

在 **主节点（Head）** 上运行：

```bash
# 启动 Ray Head 节点
ray start --head --port=6379 \
    --resources='{"aggregator_node": 1}' \
    --dashboard-host=0.0.0.0
```

- `--resources='{"aggregator_node": 1}'`：声明该节点拥有 `aggregator_node` 自定义资源，用于将 Aggregator Actor 调度到此节点。
- `--dashboard-host=0.0.0.0`：开放 Ray Dashboard（默认端口 8265）。

### 启动 Ray Worker 节点

在每个 **Worker（Client）节点** 上运行：

```bash
# Client 节点 1
ray start --address='<HEAD_IP>:6379' \
    --resources='{"client_node_1": 1}'

# Client 节点 2
ray start --address='<HEAD_IP>:6379' \
    --resources='{"client_node_2": 1}'
```

- 将 `<HEAD_IP>` 替换为 Head 节点的实际 IP。
- `client_node_1` / `client_node_2` 是自定义资源标签，用于将不同的 Client Actor 精确调度到指定机器。
- 如需更多客户端，依次增加 `client_node_3`、`client_node_4` 等。

> **单机多卡部署**：如果所有 GPU 在同一台机器上，只需在 Head 节点上同时声明所有资源：
> ```bash
> ray start --head --port=6379 \
>     --resources='{"aggregator_node": 1, "client_node_1": 1, "client_node_2": 1}'
> ```

### 验证集群状态

```bash
# 查看集群节点
ray status

# 或通过 Python 验证
python -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"
```

确认输出中包含：
- `GPU: N`（N 为集群总 GPU 数）
- `aggregator_node: 1.0`
- `client_node_1: 1.0`
- `client_node_2: 1.0`

---

## 数据准备

### 数据格式

数据文件为 **JSONL** 格式（每行一个 JSON 对象），包含以下字段：

```json
{
    "instruction": "You are an expert medical assistant named Alice...",
    "input": "Encountering a medical inquiry alongside several alternatives, your mission is to...",
    "output": "<think></think>\nAnswer: D."
}
```

| 字段 | 说明 |
|------|------|
| `instruction` | 系统提示词（System Prompt），定义模型角色 |
| `input` | 用户输入（医学问题 + 选项） |
| `output` | 期望输出（可包含 `<think>...</think>` 推理过程 + 最终答案） |

> **训练时只监督最终答案**：`train_fed_jt.py` 中的 `process_func` 会提取 `</think>` 标签之后的内容作为训练目标（labels），思维链部分被 mask 为 `-100`。

### 数据集路径配置

在 `main_new.py` 的 `dataset_map` 中配置每个 Client 的数据路径：

```python
dataset_map = {
    1: "/exp1/FedLora/data/MedQA_EN.jsonl",   # Client 1 使用英文数据
    2: "/exp1/FedLora/data/MedQA_CN.jsonl",   # Client 2 使用中文数据
}
```

各 Client 的数据 **互不共享**，这正是联邦学习的核心——数据不出域。

---

## 模型准备

### 基座模型

当前默认使用 **Qwen3-1.7B**，模型路径在 `main_new.py` 中配置：

```python
model_name = "/exp1/Qwen3/Qwen3-1.7B/"
```

> **重要**：确保该路径在所有 Ray 节点上均可访问（共享存储或每个节点都预先下载）。

### LoRA 配置

在 `train_fed_jt.py` 中定义，默认参数为：

| 参数 | 值 | 说明 |
|------|-----|------|
| `r` | 8 | LoRA 秩 |
| `lora_alpha` | 32 | 缩放因子 |
| `lora_dropout` | 0.1 | Dropout 率 |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | 应用 LoRA 的模块 |
| `task_type` | `CAUSAL_LM` | 因果语言模型 |

---

## 配置说明

`main_new.py` 中的关键训练参数：

```python
steps_per_round = 50                   # 每轮每个 Client 训练的步数
per_device_train_batch_size = 1        # 单卡 batch size
grad_accum = 4                         # 梯度累积步数（等效 batch = 1×4 = 4）
learning_rate = 3e-5                   # 学习率
logging_steps = 10                     # 每 N 步记录一次日志
ROUNDS = 2                             # 联邦训练轮数
```

### 环境变量

`main_new.py` 中的 `COMMON_ENV` 会通过 Ray 的 `runtime_env` 注入到各 Client：

```python
COMMON_ENV = {
    # --- NVIDIA 环境下以下两项可删除或留空 ---
    # "MACA_PATH": "/opt/maca",                  # 沐曦硬件专用，NVIDIA 环境不需要
    # "LD_LIBRARY_PATH": "/opt/maca/lib:...",     # 沐曦硬件专用，NVIDIA 环境不需要
    
    # --- 按需保留 ---
    "SWANLAB_PROJECT": "deepseek-1_5B-lora_sft",  # SwanLab 项目名
    "SWANLAB_API_KEY": "<your-api-key>",           # SwanLab API Key
}
```

> ⚠️ **NVIDIA 环境迁移提示**：原代码中的 `MACA_PATH`、`/opt/maca/lib` 等路径是沐曦（MetaX）GPU 的 SDK 路径。在 NVIDIA 环境中，CUDA 路径通常由系统自动管理（`/usr/local/cuda`），**无需额外设置**。建议清理这些历史环境变量以避免混淆。

### SwanLab 日志上报

如需使用 [SwanLab](https://swanlab.cn/) 进行训练可视化：

1. 在 `COMMON_ENV` 中设置 `SWANLAB_PROJECT` 和 `SWANLAB_API_KEY`。
2. Client 构造时 `report_to=("swanlab",)`（默认已开启）。
3. 如不需要，改为 `report_to=("none",)`。

---

## 运行联邦训练

### 1. 确保 Ray 集群已启动

```bash
ray status   # 确认所有节点在线，资源充足
```

### 2. 启动训练

在 **Head 节点** 上执行：

```bash
cd /exp1/FedLora
python main_new.py
```

### 3. 预期输出

```
=== Round 0 ===
分发参数数量: 0（冷启动：首轮下发为空的 LoRA 字典）
[Client 1] host=node01 CVD=0 cuda=True ndev=1 -> device=cuda:0
[Client 1] GPU=NVIDIA A100-SXM4-80GB
[Client 2] host=node02 CVD=0 cuda=True ndev=1 -> device=cuda:0
...
Client#1 -> steps(+50), global_step=50, samples=10178
Client#2 -> steps(+50), global_step=50, samples=XXXX
[Aggregator] 轮次 1 聚合完成：clients=2, keys=XX, mean=X.XXXXe-XX
聚合参数均值: X.XXXXe-XX（XX tensors）

=== Round 1 ===
分发参数数量: XX, 样例参数 dtype=torch.float32, shape=torch.Size([...])
...
[main] 所有轮次完成。
```

### 4. Checkpoint 输出

每个 Client 的训练 Checkpoint 保存在：

```
/home/fedllm/fed_ckpts/client{CID}/round-{ROUND_ID}/
```

该路径可通过 `make_trainer_for_steps` 的 `base_ckpt_dir` 参数修改。

---

## 单机训练（可选）

如只需单机 LoRA 微调（不走联邦流程），可直接运行 `train_fed_jt.py`：

```bash
python train_fed_jt.py \
    --model_name /exp1/Qwen3/Qwen3-1.7B/ \
    --dataset_name /exp1/FedLora/data/MedQA_EN.jsonl \
    --output ./lora_output/medqa_EN/Qwen3-1.7B \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 50 \
    --run_name qwen3-medqa-EN
```

支持从已有 adapter 继续训练：

```bash
python train_fed_jt.py \
    --model_name /exp1/Qwen3/Qwen3-1.7B/ \
    --adapter_path ./lora_output/medqa_EN/Qwen3-1.7B/checkpoint-100 \
    --dataset_name /exp1/FedLora/data/MedQA_EN.jsonl \
    --output ./lora_output/medqa_EN/Qwen3-1.7B
```

---

## 关键模块详解

### `main_new.py` — 联邦训练主入口

- 初始化 Ray 集群连接
- 创建 Aggregator 和多个 Client 的 Ray Actor
- 执行联邦训练主循环：**下发 → 本地训练 → 聚合 → 下一轮**

### `client_new.py` — 联邦客户端（Ray Actor）

- `_ensure_ready()`：懒加载模型、tokenizer、数据集（首次调用时初始化）
- `process_parameters(merged_lora, round_id)`：
  1. 接收聚合的全局 LoRA 并加载到本地模型
  2. 调用 `make_trainer_for_steps()` 构建 Trainer，训练 `steps_per_round` 步
  3. 提取 LoRA 权重 (`"lora_"` 关键字过滤) 回传给 Aggregator
- **优化器跨轮持久化**：`self.optimizer` 和 `self.lr_scheduler` 在轮次间保留

### `aggregator.py` — 聚合器（Ray Actor）

- `distribute_parameters()`：将当前全局 LoRA 下发（首轮为空字典 → 冷启动）
- `aggregate(client_updates)`：对所有 Client 的 LoRA 权重按样本数加权平均
  - 输入：`[{"lora_state": {...}, "num_samples": N}, ...]`
  - 输出：更新后的全局 LoRA state dict

### `train_fed_jt.py` — 训练核心逻辑

- `build_components()`：组装 tokenizer + LoRA 模型 + 处理好的训练数据集
- `make_trainer_for_steps()`：构建限定步数的 HuggingFace Trainer
- `process_func()`：数据预处理，使用 Qwen3 对话模板拼接 prompt，**只对 `</think>` 后的最终答案计算 loss**
- 支持单机独立运行（有 `main()` 函数和 `argparse`）

### `conversation_COT.py` — 对话模板

提供多种对话模板（Qwen3、LLaMA3、Vicuna 等），支持：
- Chain-of-Thought（`<think>...</think>` 标签）
- 多轮对话
- 医学专用 System Prompt

---

## 常见问题

### Q: 启动报错 `No available resources for actor`

确认 Ray 集群资源声明正确：
```bash
ray status
# 检查是否有 aggregator_node, client_node_1, client_node_2 等自定义资源
```

### Q: Client 报 `CUDA not available`

- 确认 Worker 节点的 NVIDIA 驱动和 CUDA 正常：`nvidia-smi`
- 确认 Ray Worker 启动时能检测到 GPU：`ray status` 中应显示 `GPU: N`

### Q: 模型加载失败 `FileNotFoundError`

确认模型路径在 **所有节点** 上可访问。建议使用共享文件系统（NFS / 共享存储）。

### Q: `MACA_PATH` 相关报错

这是沐曦硬件环境的残留配置。在 NVIDIA 环境中，清理 `main_new.py` 里的 `COMMON_ENV`：

```python
COMMON_ENV = {
    # 只保留需要的环境变量
    "SWANLAB_PROJECT": "your-project-name",
    "SWANLAB_API_KEY": "your-api-key",
}
```

### Q: 如何增加更多客户端？

1. 启动新的 Ray Worker 并声明 `client_node_N` 资源
2. 在 `main_new.py` 的 `dataset_map` 中添加对应数据路径
3. 修改 `for i in range(1, N+1)` 扩大客户端范围

### Q: 如何修改 LoRA 目标层？

在 Client 创建时传入 `lora_target_modules` 参数：
```python
client = FedClient.options(...).remote(
    ...
    lora_target_modules=["q_proj", "v_proj"],  # 只微调 Q 和 V
)
```

---

## 硬件环境说明

| 环境 | 状态 | 说明 |
|------|------|------|
| 沐曦（MetaX）MACA | ❌ 已弃用 | 项目早期开发环境，`utils/test_maca.py` 和 `MACA_PATH` 为历史遗留 |
| NVIDIA CUDA | ✅ 当前使用 | 标准 NVIDIA GPU + CUDA 环境，无需额外 SDK 配置 |

迁移清单：
- [x] 模型路径从 DeepSeek-R1-Distill-Qwen-1.5B 更新为 Qwen3-1.7B
- [ ] 清理 `COMMON_ENV` 中的 MACA 相关环境变量
- [ ] 更新 `aggregator.py` 中的 `model_path` 为实际使用的模型路径
- [ ] 确认 `base_ckpt_dir` 路径 (`/home/fedllm/fed_ckpts`) 在当前环境中可写

---

## License

本项目仅供学术研究使用。
