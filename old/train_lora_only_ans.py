# 新文件：COT_src/train_lora_fed_medmcqa.py
# 基于train_lora_only_answer.py修改

import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import os
import swanlab
from conversation_COT import get_conv_template, Conversation, register_conv_template, SeparatorStyle
import argparse
from functools import partial

# 环境设置
os.environ["SWANLAB_PROJECT"]="qwen3-1_7B-sft-medqa-EN"
MAX_LENGTH = 2048

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B")
    parser.add_argument("--adapter_path", type=str, default=None, help="从已有adapter继续训练的路径")
    parser.add_argument("--output", type=str, default="./lora_output/medqa_EN/Qwen3-1.7B")
    parser.add_argument("--dataset_name", type=str, default = '/home/fedllm/data/MedQA_EN.jsonl', help="MedMCQA数据集路径")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--run_name", type=str, default="qwen3-medmcqa")
    return parser.parse_args()

def process_data(dataset_path):
    """处理MedMCQA数据集"""
    messages = []
    
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            # MedMCQA数据集已经是instruction格式，直接使用
            messages.append({
                "instruction": data.get("instruction", ""),
                "input": data.get("input", ""),
                "output": data.get("output", "")
            })
    
    return messages

def get_final_answer_span(output: str) -> str:
    """提取 <think>...</think> 之后的内容"""
    end_tag = "</think>"
    if end_tag in output:
        return output.split(end_tag, 1)[1].strip()  # 取后面部分
    else:
        # fallback：如果没有 think 标签，返回整个 output
        return output.strip()

def process_func(example, tokenizer):
    """处理MedMCQA数据的预处理函数（不依赖 offset_mapping）"""
    conv = get_conv_template("qwen3_medical_en")
    
    if example['instruction']:
        system_msg = example['instruction']
    else:
        system_msg = "You are an expert medical assistant. Please provide helpful and accurate responses."
    
    conv.system_message = system_msg
    conv.append_message("user", example['input'])
    
    # === 提取最终答案内容 ===
    raw_output = example['output']
    final_content = get_final_answer_span(raw_output)
    if not final_content:
        final_content = raw_output.strip()  # fallback

    # === 构造 prompt 的两部分 ===
    # 第一部分：不含 assistant 的 prompt
    prefix_prompt = conv.get_prompt()  # user 结束后的 prompt

    # 第二部分：assistant 的输出（只保留 final_content）
    assistant_prompt = f"<|im_start|>assistant\n{final_content}<|im_end|>"

    # === 分别 tokenize ===
    # 注意：add_special_tokens 只在完整拼接时加一次
    full_prompt = prefix_prompt + assistant_prompt

    # tokenize 整个 prompt
    tokenized = tokenizer(full_prompt, add_special_tokens=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # === 构造 labels：前半 -100，后半 = input_ids ===
    labels = [-100] * len(input_ids)

    # tokenize 前缀（不含 assistant），用于确定起始位置
    prefix_tokens = tokenizer(prefix_prompt, add_special_tokens=True).input_ids
    prefix_len = len(prefix_tokens)

    # 从 prefix_len 开始，后面的 token 都是 final_content，设为 label
    for i in range(prefix_len, len(input_ids)):
        labels[i] = input_ids[i]

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    args = parse_args()
    
    print(f"加载模型: {args.model_name}")
    print(f"数据集: {args.dataset_name}")
    print(f"输出目录: {args.output}")
    
    # 加载分词器和模型
    print(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    model.enable_input_require_grads()

    # LoRA配置
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
    )

    # 关键：支持从已有adapter继续训练
    if args.adapter_path and os.path.exists(args.adapter_path):
        print(f"从已有adapter继续训练: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path,is_trainable=True)
    else:
        print("从基础模型开始新的LoRA训练")
        model = get_peft_model(model, config)

    model.print_trainable_parameters()

    # 数据处理
    print(f"处理数据集: {args.dataset_name}")
    messages = process_data(args.dataset_name)
    print(f"数据集大小: {len(messages)}")
    
    # 创建训练数据集
    df = pd.DataFrame(messages)
    train_ds = Dataset.from_pandas(df)
    
    # 使用partial传递tokenizer参数
    process_func_with_tokenizer = partial(process_func, tokenizer=tokenizer)
    
    train_dataset = train_ds.map(
        process_func_with_tokenizer,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train data",
        batched=False,
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        optim="adamw_torch",
        seed=42,
        fp16=False,
        bf16=True,
        report_to=["swanlab"],
        run_name=args.run_name,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # 开始训练
    print("开始训练...")
    trainer.train()
    

if __name__ == "__main__":
    main()