
import ray
import torch
from transformers import AutoModelForCausalLM
from typing import Dict, List
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.storage")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.load")

model_path = "/home/fedllm/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B"

@ray.remote(num_gpus=1, resources={"aggregator_node": 1})
class Aggregator:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True
        )
        self.target_params = {
            n: p for n, p in self.model.named_parameters()
            if "q_proj" in n or "v_proj" in n
        }
        self.global_lora_state: Dict[str, torch.Tensor] = {}
        self.round = 0

    def distribute_parameters(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().float() for k, v in self.global_lora_state.items()}
        # return {n: p.detach().cpu().half() for n, p in self.target_params.items()}

    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        期望的 client_updates:
          [
            {"lora_state": {k: Tensor, ...}, "num_samples": int},
            ...
          ]
        """
        if not client_updates:
            print("[Aggregator] 未收到更新，继续分发上轮参数")
            return self.distribute_parameters()

        sum_states = defaultdict(lambda: None)   # k -> 累加的加权和
        weight_sum = defaultdict(float)          # k -> 权重和

        total_clients = 0
        for upd in client_updates:
            lora_sd = upd.get("lora_state", {})
            n = float(upd.get("num_samples", 1))
            total_clients += 1

            for k, t in lora_sd.items():
                x = t.detach().cpu().float()
                if sum_states[k] is None:
                    sum_states[k] = x * n
                else:
                    sum_states[k].add_(x, alpha=n)
                weight_sum[k] += n

        new_global = {}
        for k, s in sum_states.items():
            w = max(weight_sum[k], 1e-9)
            new_global[k] = (s / w).contiguous()

        self.global_lora_state = new_global
        self.round += 1

        if new_global:
            means = [v.mean().item() for v in new_global.values()]
            print(f"[Aggregator] 轮次 {self.round} 聚合完成：clients={total_clients}, keys={len(new_global)}, mean={sum(means)/len(means):.4e}")
        else:
            print(f"[Aggregator] 轮次 {self.round} 聚合完成，但为空（可能客户端未回 lora_state）")

        # 返回可直接下发的全局 LoRA
        return self.distribute_parameters()

    # def aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    #     avg_updates = {}
    #     print(f"[Aggregator] 收到 {len(client_updates)} 个客户端更新")
    #     for key in client_updates[0].keys():
    #         try:
    #             stacked = torch.stack([update[key].float().cpu() for update in client_updates])
    #             avg_updates[key] = stacked.mean(dim=0)
    #         except Exception as e:
    #             print(f"[Aggregator] 聚合 {key} 时出错：{e}")

    #     # 更新到模型中
    #     with torch.no_grad():
    #         for name, param in self.model.named_parameters():
    #             if name in avg_updates:
    #                 param.copy_(avg_updates[name].to(param.device, dtype=param.dtype))
    #             # else:
    #             #     # 可以选择跳过或警告
    #             #     # print(f"[Aggregator] 参数 {name} 未收到客户端更新，跳过")

    #     return avg_updates

