
import ray
import torch
from transformers import AutoModelForCausalLM
from typing import Dict, List
import warnings

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

    def distribute_parameters(self) -> Dict[str, torch.Tensor]:
        return {n: p.detach().cpu().half() for n, p in self.target_params.items()}

    def aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        avg_updates = {}
        print(f"[Aggregator] 收到 {len(client_updates)} 个客户端更新")
        for key in client_updates[0].keys():
            try:
                stacked = torch.stack([update[key].float().cpu() for update in client_updates])
                avg_updates[key] = stacked.mean(dim=0)
            except Exception as e:
                print(f"[Aggregator] 聚合 {key} 时出错：{e}")

        # 更新到模型中
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in avg_updates:
                    param.copy_(avg_updates[name].to(param.device, dtype=param.dtype))
                else:
                    # 可以选择跳过或警告
                    print(f"[Aggregator] 参数 {name} 未收到客户端更新，跳过")

        return avg_updates

