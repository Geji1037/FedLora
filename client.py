

import ray
import torch
import socket
from typing import Dict
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training,set_peft_model_state_dict
from train_utils import train_one_step

from torch.utils.data import DataLoader
from transformers import default_data_collator

@ray.remote
class Client:
    def __init__(self, cid: int,model_path:str):
        self.cid = cid
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        has_cuda = torch.cuda.is_available()
        ndev = torch.cuda.device_count()   
        self.device = torch.device("cuda:0" if has_cuda and ndev > 0 else "cpu")
        self.hostname = socket.gethostname()
        self.model_path = model_path
        print(f"[Client {self.cid}] host={self.hostname} "
              f"CVD={cvd} cuda_avail={has_cuda} ndev={ndev} -> device={self.device}")
        if has_cuda and ndev > 0:
            try:
                print(f"[Client {self.cid}] GPU_NAME={torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"[Client {self.cid}] GPU_NAME_ERR={e}")
        self.model = None
        self.tokenizer = None

    def load_model(self,param_template:Dict[str,torch.Tensor]):
        print(f"[Client {self.cid}] 初始化模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True
        )

        # 用 LoRA 包装模型（注意要配置正确的 target modules）
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(base_model, lora_config)
        model = model.to(self.device)
        model.train()
        self.model = model

        if param_template and any("lora_" in k for k in param_template.keys()):
            # 将聚合来的 LoRA 写入本地 LoRA
            lora_loaded_missing, lora_unexpected = set_peft_model_state_dict(
                self.model,
                {k: v.to(self.device, dtype=self.model.dtype) for k, v in param_template.items()},
                adapter_name="default"
            )
            matched = len(param_template) - len(lora_loaded_missing)
            print(f"[Client {self.cid}] 加载聚合LoRA：匹配{matched} 缺失{len(lora_loaded_missing)} 意外{len(lora_unexpected)}")
        else:
            # 兼容旧路径（如果还是发来 q_proj/v_proj 的全量矩阵，就按旧逻辑覆盖）
            loaded, missing = 0, []
            for name, param in model.named_parameters():
                if name in param_template:
                    param.data = param_template[name].to(self.device).to(param.dtype)
                    loaded += 1
                elif "lora_" in name:
                    missing.append(name)
            print(f"[Client {self.cid}] 覆盖参数数：{loaded}；未匹配的 LoRA 参数样例：{missing[:5]}")


        # 打印一下结构
        # print(model)

        # missing ,loaded = [],0

        # # 将来自 Aggregator 的权重加载到 model 中
        # for name, param in model.named_parameters():
        #     if name in param_template:
        #         param.data = param_template[name].to(self.device).to(param.dtype)
        #         loaded += 1
        #     elif "lora_" in name:
        #         missing.append(name)
        # print(f"[Client {self.cid}] 覆盖参数数：{loaded}；未匹配的 LoRA 参数样例：{missing[:5]}")

    def ping(self):
        return {"cid": self.cid, "host": self.hostname, "device": str(self.device)}
        pass

    def process_parameters(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """接收参数，加载模型，执行本地训练，并返回更新后的权重"""
        self.load_model(params)

        # 👉 执行一轮训练（可以放到 train_utils.py 中）
        train_one_step(self.model, self.tokenizer, self.device)

        # 返回更新后的 LoRA 权重
        updated_params = {
            n: p.detach().cpu().half()
            for n, p in self.model.state_dict().items()
            if "lora_" in n and n.endswith("weight")
        }

        return {"lora_state": updated_params, "num_samples": 1024}
        # updated_params = {
        #     n: p.detach().cpu().half() for n, p in self.model.named_parameters()
        #     if "lora_" in n  # 只发送 LoRA 权重
        # }

        # return updated_params
