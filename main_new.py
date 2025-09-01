# main.py
import os
import ray
import numpy as np

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
    model_name = "/home/fedllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B"
    # 按客户端划分数据集（示例）
    dataset_map = {
        1: "/home/fedllm/MedQA_EN.jsonl",
        2: "/home/fedllm/MedQA_CN.jsonl",
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
    # 你原来是 range(1,3)，即客户端 1 和 2
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
        if not params:
            print("分发参数数量: 0（冷启动：首轮下发为空的 LoRA 字典）")
        else:
            any_tensor = next(iter(params.values()))
            print(f"分发参数数量: {len(params)}, 样例参数 dtype={any_tensor.dtype}, shape={getattr(any_tensor, 'shape', None)}")

        # 客户端训练（每个客户端内部已调用 Trainer 按 steps_per_round 进行）
        futures = [c.process_parameters.remote(params,round_id=rd) for c in Clients]
        client_results = ray.get(futures)

        # 打印每个客户端的简报
        for r in client_results:
            rep = r.get("report", {})
            print(f"Client#{rep.get('client_id')} -> steps(+{rep.get('round_steps')}), "
                  f"global_step={rep.get('global_step')}, samples={rep.get('num_samples')}")

        # 聚合 LoRA（加权策略在 Aggregator 内部）
        global_update = ray.get(aggregator.aggregate.remote(client_results))
        if not global_update:
            print("[WARN] 聚合返回空字典，请检查 aggregator.aggregate 实现")
        else:
            means = [t.mean().item() for t in global_update.values() if hasattr(t, "mean")]
            print(f"聚合参数均值: {np.mean(means):.4e}（{len(means)} tensors）")

    print("\n[main] 所有轮次完成。")

if __name__ == "__main__":
    main()
