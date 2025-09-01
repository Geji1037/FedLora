import ray
import numpy as np
from aggregator import Aggregator
from client import Client
from train_utils import train_one_step

import os

COMMON_ENV = {
    "MACA_PATH": "/opt/maca",
    "LD_LIBRARY_PATH": "/opt/maca/lib:/opt/maca/lib64:/usr/local/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
    "PATH": "/opt/maca/bin:" + os.environ.get("PATH", "")
}

def main():
    ray.init(      
        address='auto',      
        runtime_env={
                "working_dir":".",
                "env_vars":{"PYTHONPATH":"."} ,
                "excludes":[
                    "models","models/**"
                ]
            }
            )

    aggregator = Aggregator.remote()
    Clients = []
    model_path = "/home/fedllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B"
    for i in range(1,3):
        resource_key = f"client_node_{i}"
        resource_value = 1
        resources = {resource_key:resource_value}
        client = Client.options(
            num_gpus=1,
            resources=resources,
            runtime_env={"env_vars":COMMON_ENV}
        ).remote(i,model_path=model_path)
        Clients.append(client)    
    ray.get([client.ping.remote() for client in Clients])

    for round in range(3):
        print(f"\n=== Round {round} ===")

        params = ray.get(aggregator.distribute_parameters.remote())
        print(f"分发参数数量: {len(params)}, 样例参数形状: {next(iter(params.values())).shape}")

        futures = [client.process_parameters.remote(params) for client in Clients]
        client_results = ray.get(futures)

        global_update = ray.get(aggregator.aggregate.remote(client_results))
        print(f"聚合参数均值: {np.mean([u.mean().item() for u in global_update.values()]):.4e}")

if __name__ == "__main__":
    main()