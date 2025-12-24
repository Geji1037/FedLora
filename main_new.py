# main.py
import os
import ray
import numpy as np
import time ## 用于计时

from aggregator import Aggregator
# ✅ 改成从 client_new 导入
from client_new import Client as FedClient

# 你原本的可执行环境变量
COMMON_ENV = {
    "MACA_PATH": "/opt/maca",
    "LD_LIBRARY_PATH": "/opt/maca/lib:/opt/maca/lib64:/usr/local/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
    "PATH": "/opt/maca/bin:" + os.environ.get("PATH", ""),
    # 需要上报到 SwanLab 则可在这里设置（可选）
    "SWANLAB_PROJECT": "deepseek-1_5B-lora_sft",
    "SWANLAB_API_KEY": "pUDnewWbjDrd1iRWqP4jS"
}


def summarize_model_payload(model_dict):
    """
    model_dict: Dict[str, torch.Tensor]
    返回模型大小、dtype 分布等信息
    """
    if not model_dict:
        return {
            "num_tensors": 0,
            "total_bytes": 0,
            "total_mb": 0.0,
            "dtypes": {},
        }

    total_bytes = 0
    dtypes = {}

    for t in model_dict.values():
        if not hasattr(t, "numel"):
            continue
        numel = t.numel()
        elem_size = t.element_size()  # bytes per element
        total_bytes += numel * elem_size

        dtype = str(t.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + 1

    return {
        "num_tensors": len(model_dict),
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 ** 2),
        "dtypes": dtypes,
    }


def main():
    ray.init(
        address="auto",
        runtime_env={
            "working_dir": ".",
            "env_vars": {"PYTHONPATH": "."},
            "excludes": ["models", "models/**"],  # 避免把大模型目录打包到 Ray
        },
    )

    # -----------------------
    # 全局配置（按需改）
    # -----------------------
    model_name = "/exp1/Qwen3/Qwen3-1.7B"
    # 按客户端划分数据集（示例）
    dataset_map = {
        1: "/exp1/MedQA_EN.jsonl",
        2: "/exp1/MedQA_CN.jsonl",
        # 3: "/home/fedllm/data/MedQA_CN.jsonl",
    }
    steps_per_round = 50
    per_device_train_batch_size = 1
    grad_accum = 4
    learning_rate = 3e-5
    logging_steps = 10

    # -----------------------
    # 初始化聚合器与客户端
    # -----------------------
    aggregator = Aggregator.remote()

    Clients = []

    for i in range(1, 3):
        resource_key = f"client_node_{i}"
        resources = {resource_key: 1}

        data_path = dataset_map.get(i)
        if data_path is None:
            raise ValueError(f"[main] 缺少客户端 {i} 的数据路径！请在 dataset_map 中补全。")

        # ✅ 用新的 FedClient + 新构造参数
        client = FedClient.options(
            num_gpus=1,
            resources=resources,
            runtime_env={"env_vars": COMMON_ENV},
        ).remote(
            cid=i,
            model_name=model_name,
            dataset_path=data_path,
            adapter_path=None,  # 如需从已有 LoRA 继续，填路径
            steps_per_round=steps_per_round,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=grad_accum,
            logging_steps=logging_steps,
            lora_target_modules=None,      # 默认使用 train_lora_only_ans.py 中的 target 列表
            report_to=("swanlab",),           # 上报到 SwanLab 则改成 ("swanlab",)
        )
        Clients.append(client)

    # 心跳
    _ = ray.get([c.ping.remote() for c in Clients])

    # -----------------------
    # 联邦训练主循环
    # -----------------------
    ROUNDS = 2
    for rd in range(ROUNDS):
        print(f"\n=== Round {rd} ===")

        # 下发当前全局 LoRA（可能为空：冷启动）
        
        params = ray.get(aggregator.distribute_parameters.remote())
        t_cast_start = time.perf_counter()
        payload_info = summarize_model_payload(params)
        print(
            f"[Round {rd}] Broadcast payload: "
            f"{payload_info['num_tensors']} tensors, "
            f"{payload_info['total_mb']:.2f} MB, "
            f"dtypes={payload_info['dtypes']}"
        )
        if not params:
            print("分发参数数量: 0（冷启动：首轮下发为空的 LoRA 字典）")
        else:
            any_tensor = next(iter(params.values()))
            print(f"分发参数数量: {len(params)}, 样例参数 dtype={any_tensor.dtype}, shape={getattr(any_tensor, 'shape', None)}")

        ack_futures = {i:Clients[i-1].ack_params.remote(params,round_id=rd) for i in range(1,3)}

        cast_per_client = {}
        for cid,fut in ack_futures.items():
            _ = ray.get(fut)
            cast_per_client[cid] = time.perf_counter() - t_cast_start

        t_cast_s = max(cast_per_client.values())
        print(f"[Round {rd}] per-client broadcast latency  : {cast_per_client} s")
        print(f"[Round {rd}] max broadcast latency         : {t_cast_s:.3f} s")

        ack_futures = [c.ack_params.remote(params,round_id=rd) for c in Clients]
        _ = ray.get(ack_futures)

        t_cast_end = time.perf_counter()
        t_cast_s = t_cast_end - t_cast_start

        # 客户端训练（每个客户端内部已调用 Trainer 按 steps_per_round 进行）
        futures = [c.process_parameters.remote(params,round_id=rd) for c in Clients]

        
        t_ul_start = time.perf_counter()

        futures = {cid:c.process_parameters.remote(params,round_id=rd) for cid,c in enumerate(Clients,start=1)}
        recv_per_client = {}
        results = {}
        for cid, fut in futures.items():
            r = ray.get(fut)
            results[cid] = r
            recv_per_client[cid] = time.perf_counter() - t_ul_start 

        upload_per_client = {}
        for cid,r in results.items():
            rep = r.get("report", {})
            train_t = rep.get("train_time_sec", 0.0)
            pack_t = rep.get("pack_time_sec", 0.0)
            upload_per_client[cid] = recv_per_client[cid] - train_t - pack_t

        print(f"[Round {rd}] per-client receive latency  : {recv_per_client} s")
        print(f"[Round {rd}] per-client upload latency   : {upload_per_client} s")

        t_ul_end = time.perf_counter()
        clients_total_time_s = t_ul_end - t_ul_start

        client_results = list(results.values())
        # 打印每个客户端的简报
        for r in client_results:
            rep = r.get("report", {})
            print(f"Client#{rep.get('client_id')} -> steps(+{rep.get('round_steps')}), "
                  f"global_step={rep.get('global_step')}, samples={rep.get('num_samples')}")

        train_times = [
            r["report"].get("train_time_sec", 0.0)
            for r in client_results
        ]
        max_train_time = max(train_times)

        t_ul_wait_s = clients_total_time_s - max_train_time

        

        # 聚合 LoRA（加权策略在 Aggregator 内部）
        agg_ret = ray.get(aggregator.aggregate.remote(client_results))

        global_update = agg_ret["merged_lora"]
        agg_time_s = agg_ret["agg_time_s"]

        payload_info = summarize_model_payload(global_update)
        print(
            f"[Round {rd}] Aggregated model payload: "
            f"{payload_info['num_tensors']} tensors, "
            f"{payload_info['total_mb']:.2f} MB, "
            f"dtypes={payload_info['dtypes']}"
        )


        if not global_update:
            print("[WARN] 聚合返回空字典，请检查 aggregator.aggregate 实现")
        else:
            means = [t.mean().item() for t in global_update.values() if hasattr(t, "mean")]
            print(f"聚合参数均值: {np.mean(means):.4e}（{len(means)} tensors）")

    # ===============================
    # Round-level timing summary
    # ===============================
        round_time_s = (
            t_cast_s
            + clients_total_time_s
            + agg_time_s
        )

        print(
            f"\n[Round {rd}] === Timing Summary ===\n"
            f"  Broadcast time (t_cast)      : {t_cast_s:.3f} s\n"
            f"  Client total time            : {clients_total_time_s:.3f} s\n"
            f"    ├─ Max client train time   : {max_train_time:.3f} s\n"
            f"    └─ Upload/Wait time        : {t_ul_wait_s:.3f} s\n"
            f"  Aggregation compute time     : {agg_time_s:.3f} s\n"
            f"  ----------------------------------------\n"
            f"  Estimated round time         : {round_time_s:.3f} s\n"
        )

    print("\n[main] 所有轮次完成。")

if __name__ == "__main__":
    main()
