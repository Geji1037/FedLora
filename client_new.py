# client_new.py
import os
import ray
import torch
import socket
from typing import Dict, Optional, Tuple

from peft import set_peft_model_state_dict
from train_fed_jt import build_components, make_trainer_for_steps


@ray.remote
class Client:
    def __init__(
        self,
        cid: int,
        model_name: str,
        dataset_path: str,
        adapter_path: Optional[str] = None,
        steps_per_round: int = 50,
        learning_rate: float = 3e-5,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        logging_steps: int = 10,
        lora_target_modules: Optional[list] = None,
        report_to: Tuple[str, ...] = ("swanlab",),  # 需要上报到 swanlab 就填 ("swanlab",)
    ):
        self.cid = cid
        self.hostname = socket.gethostname()

        # 基本参数
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.adapter_path = adapter_path
        self.steps_per_round = steps_per_round
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.logging_steps = logging_steps
        self.lora_target_modules = lora_target_modules
        self.report_to = report_to

        # 运行时对象
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0

        # 显卡信息
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        has_cuda = torch.cuda.is_available()
        ndev = torch.cuda.device_count()
        device = torch.device("cuda:0" if has_cuda and ndev > 0 else "cpu")
        self.device = device
        print(f"[Client {self.cid}] host={self.hostname} CVD={cvd} cuda={has_cuda} ndev={ndev} -> device={device}")
        if has_cuda and ndev > 0:
            try:
                print(f"[Client {self.cid}] GPU={torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"[Client {self.cid}] GPU_NAME_ERR={e}")

    # -- 懒加载：首次构建 tokenizer/model/dataset --
    def _ensure_ready(self):
        if self.model is not None:
            return
        self.tokenizer, self.model, self.train_dataset = build_components(
            self.model_name,
            self.dataset_path,
            adapter_path=self.adapter_path,
            lora_target_modules=self.lora_target_modules,
        )
        self.model.train()

    def ping(self):
        return {"cid": self.cid, "host": self.hostname, "device": str(self.device)}

    # -- 每轮：接收聚合 LoRA -> 训练若干步 -> 回传 LoRA --
    def process_parameters(self, merged_lora: Optional[Dict[str, torch.Tensor]] = None,round_id: int = 0):
        self._ensure_ready()

        # 载入聚合来的 LoRA（只匹配存在且形状相同的键）
        if merged_lora:
            cur_sd = self.model.state_dict()
            filtered = {
                k: v.to(self.device, dtype=cur_sd[k].dtype)
                for k, v in merged_lora.items()
                if (k in cur_sd and v.shape == cur_sd[k].shape)
            }
            if filtered:
                try:
                    set_peft_model_state_dict(self.model, filtered, adapter_name="default")
                except TypeError:
                    set_peft_model_state_dict(self.model, filtered)
                print(f"[Client {self.cid}] 加载聚合LoRA：匹配 {len(filtered)}")
            else:
                print(f"[Client {self.cid}] 聚合LoRA键未匹配到，可忽略（本地继续训练）")

        # 计算“本轮训练后的总步数上限”
        target_total_steps = self.global_step + int(self.steps_per_round)

        # 组装/复用优化器与调度器（跨轮持久化）
        optimizers = (self.optimizer, self.lr_scheduler) if (self.optimizer and self.lr_scheduler) else None
        trainer = make_trainer_for_steps(
            self.model,
            self.tokenizer,
            self.train_dataset,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_steps=self.logging_steps,
            run_name=f"client{self.cid}",
            client_id=f"client{self.cid}",
            round_id=round_id,
            bf16=True,
            max_steps=target_total_steps,
            report_to=self.report_to,
            optimizers=optimizers,
        )

        # 开始训练到 target_total_steps
        trainer.train()

        # 持久化优化器/调度器与 global_step
        self.optimizer, self.lr_scheduler = trainer.optimizer, trainer.lr_scheduler
        self.global_step = int(getattr(trainer.state, "global_step", target_total_steps))

        # 回传 LoRA 权重（不要只用 endswith('weight')，会漏键）
        lora_state = {
            n: p.detach().cpu().to(dtype=torch.float16)
            for n, p in self.model.state_dict().items()
            if "lora_" in n
        }

        # 可选：返回简单统计
        report = {
            "client_id": self.cid,
            "host": self.hostname,
            "round_steps": int(self.steps_per_round),
            "global_step": self.global_step,
            "num_samples": int(len(self.train_dataset)),
        }
        print(f"[Client {self.cid}] 训练完成：{report}")
        return {"lora_state": lora_state, "report": report}
