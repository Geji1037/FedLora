# Federated LoRA Training – Timing & Communication Measurement

本项目基于 **Ray + LoRA + Federated Learning** 实现了一个可运行的多客户端联邦训练系统，并在**真实系统路径**上对**计算时间与通信相关时间**进行了细粒度插桩与统计。

本 README 重点说明：

- 一轮联邦训练（round）的完整执行流程  
- 每一个时间变量的定义、代码来源与语义  
- 如何理解 “broadcast / upload / wait” 等时间  
- 本系统采用的**端到端系统级计时口径**

---

## 1. Overall Execution Flow (One Round)

一轮联邦训练（Round `b`）的执行顺序如下：

### Server:
1. Prepare global LoRA parameters  
2. Broadcast parameters to all clients  
3. Wait for clients to finish local training & upload updates  
4. Aggregate client updates into new global LoRA  

### Client `k`:
1. Receive global LoRA parameters  
2. Load LoRA into local model  
3. Run local training (`steps_per_round` steps)  
4. Extract & pack LoRA parameters  
5. Return update to server  

---

## 2. Key Timing Variables Overview

| Symbol / Name            | Meaning                                   | Where Measured     |
|--------------------------|-------------------------------------------|--------------------|
| `t_cast`                 | Broadcast latency (server → clients)      | `main.py`          |
| `t_comp(k,b)`            | Client local training time                | `client_new.py`    |
| `t_pack(k,b)`            | Client LoRA packing time                  | `client_new.py`    |
| `t_update(k,b)`          | Client update latency (end-to-end)        | `main.py`          |
| `t_uplink+sys(k,b)`      | Upload-like latency (derived)             | `main.py`          |
| `t_agg(b)`               | Server aggregation compute time           | `aggregator.py`    |
| `T_round(b)`             | Total round latency                       | `main.py`          |

---

## 3. Broadcast Time (`t_cast`)

### Definition

Broadcast time measures how long it takes for the global LoRA parameters to be delivered from the server to all clients and acknowledged.

Formally:

\[
t_{k,b}^{\text{cast}} = t_{k}^{\text{ack}} - t_{\text{cast\_start}}
\]

The round-level broadcast time is defined as:

\[
t_{b}^{\text{cast}} = \max_k t_{k,b}^{\text{cast}}
\]

### Code Location (`main.py`)

```python
t_cast_start = time.perf_counter()

params = ray.get(aggregator.distribute_parameters.remote())

ack_futures = {
    i: Clients[i-1].ack_params.remote(params, round_id=rd)
    for i in range(1, 3)
}

cast_per_client = {}
for cid, fut in ack_futures.items():
    ray.get(fut)
    cast_per_client[cid] = time.perf_counter() - t_cast_start

t_cast_s = max(cast_per_client.values())
```
### Interpretation

- Includes Ray object store transfer and network downlink  
- Does **not** include training or model computation  
- Closest approximation to pure broadcast communication latency  
- Typically very fast (sub-second for ~33MB LoRA payload)

---

## 4. Client-Side Timing (Measured at Client)

All client-side timing is measured inside `Client.process_parameters()`.

### 4.1 Client Compute Time (`t_comp`)

**Definition:**  
Time spent in local training (forward/backward/optimizer steps).

**Code:**

```python
t_train_start = time.perf_counter()
trainer.train()
t_train_end = time.perf_counter()

train_time_sec = t_train_end - t_train_start

### 4.2 Client Packing Time (`t_pack`)

**Definition:**  
Time spent extracting LoRA parameters from the model and preparing them for upload.

**Code:**

```python
t_pack_start = time.perf_counter()

lora_state = {
    n: p.detach().cpu().to(dtype=torch.float16)
    for n, p in self.model.state_dict().items()
    if "lora_" in n
}

t_pack_end = time.perf_counter()
```
### 4.3 Client Total Processing Time (t_process)
Definition:
Total time spent inside process_parameters().

```python
t_start = time.perf_counter()
...
t_end = time.perf_counter()

process_time_sec = t_end - t_start
```
## 5.Client Update Latency(t_update)
Definition:
End-to-end time from when the server dispatches a training task to when it receives the client update.
Measured on the server side, not inside the client.
```python
t_ul_start = time.perf_counter()

futures = {
    cid: c.process_parameters.remote(params, round_id=rd)
    for cid, c in enumerate(Clients, start=1)
}

recv_per_client = {}
results = {}
for cid, fut in futures.items():
    r = ray.get(fut)
    results[cid] = r
    recv_per_client[cid] = time.perf_counter() - t_ul_start
```
Interpretation:
This time includes:

- Client local training
- LoRA packing
- Ray serialization & scheduling
- Network uplink
- Server-side deserialization    
❗ It is not pure network upload time.
## 6.Upload-like Latency(t_uplink + sys)
Definition(Derived)  
Approximate uplad-related overhead by subtracting known computation components:
$$
t_{k,b}^{\text{uplink+sys}} = t_{k,b}^{\text{update}} - t_{k,b}^{\text{comp}} - t_{k,b}^{\text{pack}}
$$
```python
upload_per_client[cid] = (
    recv_per_client[cid]
    - train_time_sec
    - pack_time_sec
)
```
Important Note :  
This value:
- Is not pure network upload time
- Includes serialization, Ray runtime overhead, scheduling delays, etc.
- However, it accurately reflects the system-level cost of client updates and is suitable for performance analysis and comparison.

### 7. Server Aggregation Time (`t_agg`)

**Definition:**  
Time spent aggregating client LoRA updates on the server.

**Code Location (`aggregator.py`):**

```python
t_agg_start = time.perf_counter()
# aggregation logic
t_agg_end = time.perf_counter()

agg_time_s = t_agg_end - t_agg_start
```

### 8. Upload / Wait Time (Straggler Effect)

**Definition:**  
Measures how long the server waits for slower clients after the fastest client finishes training:

$$
t_{b}^{\text{wait}} = \max_{k} t_{k,b}^{\text{update}} - \max_{k} t_{k,b}^{\text{comp}}
$$

**Code:**

```python
t_ul_wait_s = clients_total_time_s - max_train_time
```
Interpretation:

- Captures straggler and system synchronization overhead
- Useful for analyzing heterogeneity effects across clients

### 9. Total Round Time (`T_round`)

The total duration of one federated round is approximated as:

$$
T_b = t_b^{\text{cast}} + \max_{k} t_{k,b}^{\text{update}} + t_b^{\text{agg}}
$$

**Code:**

```python
round_time_s = (
    t_cast_s
    + clients_total_time_s
    + agg_time_s
)