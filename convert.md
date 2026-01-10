


# FedLora  FedROC 架构改造演进方案

## 1. 核心架构变更对比

| 特性 | 当前架构 (FedLora) | 目标架构 (FedROC) |
| --- | --- | --- |
| **拓扑结构** | **星型 (Star)** <br>

<br> 1个中心 Server连接所有 Clients | <br>**链式/环式 (Chain/Ring)** 

<br>

<br> 多个 Edge Server (ES) 串联，通过重叠节点连接 |
| **聚合方式** | **Global Aggregation** <br>

<br> 所有梯度上传至中心聚合 | <br>**Edge Aggregation + Relay** <br>

<br> 边缘服务器局部聚合 + 重叠节点跨域中继 

 |
| **Client 角色** | 同构 (均为普通 Client) | 分级角色 ：

<br>

<br> 1. **LCs**: 本地节点 (只连1个ES)<br>

<br> 2. **ROCs**: 中继重叠节点 (连2个ES，负责转发) |
| **通信流** | Upload  Aggregate  Broadcast | Broadcast  Train  **Relay (Cross-Cluster)**  Edge Aggregate |

---

## 2. 改造实施路径 (Step-by-Step)

### 步骤一：定义拓扑与配置结构 (Configuration)

**修改文件**: `main_new.py`, `config`
**目标**: 摒弃扁平的 `client_num` 配置，建立“集群 (Cell)”概念。

你需要定义每个 Client 属于哪个边缘服务器（ES），以及谁是那个特殊的“重叠中继（ROC）”。

```python
# 伪代码：新的拓扑配置结构
TOPOLOGY_CONFIG = {
    "num_edge_servers": 3,  # 论文中的 L
    "clusters": {
        0: {"es_name": "ES_0", "local_clients": ["c1", "c2"], "relay_client": "c3"}, # c3 连接 ES_0 和 ES_1
        1: {"es_name": "ES_1", "local_clients": ["c4", "c5"], "relay_client": "c6"}, # c6 连接 ES_1 和 ES_2
        2: {"es_name": "ES_2", "local_clients": ["c7", "c8"], "relay_client": None}  # 链尾
    }
    # [cite_start]论文定义：ROC (Relay Overlapping Client) 位于两个 Cell 的重叠区 [cite: 228]
}

```

### 步骤二：重构聚合器为边缘服务器 (Edge Server)

**修改文件**: `aggregator.py`
**目标**: `Aggregator` 不再是全局单例，而是实例化多个 `EdgeServer` Actor。

1. **去中心化实例化**: 在 `main_new.py` 中循环启动多个 `EdgeServer` Actor。
2. **邻居感知**: 每个 `EdgeServer` 需要知道自己的 `neighbor_ids`（左邻/右邻），以便处理来自中继的数据。
3. **聚合逻辑升级**:
* **原逻辑**: `Avg(w_local)`
* 
**新逻辑 (FedROC Eq. 12)**:




* 你需要为 `EdgeServer` 增加一个接口 `receive_relay_weights()` 来接收邻居通过 ROC 传来的模型。



### 步骤三：客户端角色分级 (Client Roles)

**修改文件**: `client_new.py`
**目标**: 让 Client 支持“双连接”和“中继转发”。

在 `__init__` 中区分角色：

* **Local Client (LC)**: 保持现状，只持有一个 `es_handle`。
* **Relay Client (ROC)**: 持有两个 `es_handles` (Current & Neighbor)。

**ROC 的特殊行为流程**:

1. **Receive**: 从两个 ES 接收模型  和 。
2. 
**Internal Aggregate**: 在训练前，先将两个模型聚合（通常是加权平均，或者取最早到达的那个 ）。


3. **Train**: 基于聚合后的模型进行 LoRA 微调。
4. **Forward (关键)**: 训练完成后，将更新后的权重**发送给邻居 ES**，而不是仅发回原 ES。

### 步骤四：编排与控制流 (Orchestration)

**修改文件**: `main_new.py` (Coordinator)
**目标**: 控制同步/异步节奏。

论文中存在明确的时序依赖 。Ray 的异步特性很好，但需要手动控制同步点：

1. **Broadcast Phase**: Coordinator 通知所有 ES 下发参数。
2. **Local Update Phase**: 所有 Client (LCs & ROCs) 并行训练。
3. **Relay Phase (新增)**:
* 检测到 ROC 完成训练后，触发 ROC 将参数 `ray.put` 到邻居 ES 的 Object Store。
* *注意*: 普通 Client 直接上传回所属 ES。


4. **Edge Aggregation Phase**:
* 每个 ES 等待：(所有本地 Client 上传) **AND** (邻居 ROC 的中继包到达)。
* 执行聚合，更新版本。



---

## 3. 关键代码接口设计预览

### A. Client Actor (`client_new.py`)

```python
class ClientActor:
    def __init__(self, role="LC", neighbor_es_handle=None, ...):
        self.role = role # 'LC' or 'ROC'
        self.neighbor_handle = neighbor_es_handle # 仅 ROC 有此句柄

    def train(self, global_weights_primary, global_weights_secondary=None):
        # [cite_start]1. 如果是 ROC，先融合两个来源的权重 [cite: 260]
        if self.role == "ROC" and global_weights_secondary:
            init_weights = self.aggregate_edge_models(global_weights_primary, global_weights_secondary)
        else:
            init_weights = global_weights_primary
        
        # 2. 正常训练 LoRA
        new_weights = self.trainer.train(init_weights)

        # 3. 返回结果 (ROC 需要特殊标记，以便 Main 知道要转发给谁)
        return {
            "weights": new_weights,
            "role": self.role,
            "target_neighbor": self.neighbor_handle if self.role == "ROC" else None
        }

```

### B. Edge Server Actor (`aggregator.py`)

```python
class EdgeServer:
    def __init__(self, cluster_id):
        self.local_buffer = []
        [cite_start]self.relay_buffer = [] # 存放隔壁传来的模型 [cite: 297]

    def aggregate(self):
        # [cite_start]论文核心公式：本地数据 + 中继数据 [cite: 314]
        # w_new = (Sum(w_local * n_local) + Sum(w_relay * n_relay)) / Total_N
        total_weights = self.weighted_avg(self.local_buffer + self.relay_buffer)
        return total_weights

```

---

## 4. 实施阶段建议

1. **Phase 0 (数据准备)**: 确保你的 `dataset_map` 可以被切分成 3-4 份，模拟不同的 Cluster 数据分布（Non-IID）。
2. **Phase 1 (双 Server)**: 先不加 Relay，跑通 2 个独立的 Edge Server，验证 Ray 能同时跑两个聚合器。
3. **Phase 2 (加入 Relay)**: 引入 ROC 逻辑，实现“左手收模型，右手发模型”。
4. 
**Phase 3 (全链路)**: 对齐论文的 Loss 曲线，观察 ROC 是否真的加速了收敛（论文宣称能加速收敛 ）。