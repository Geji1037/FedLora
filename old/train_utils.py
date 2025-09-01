# train_utils.py
import torch
from torch.nn.utils import clip_grad_norm_

def train_one_step(model, tokenizer, device, steps: int = 16, lr: float = 5e-4, optimizer=None):
    """
    最小改动版：
    - 屏蔽 PAD（labels= -100）
    - 连续训练多步（steps）
    - 可选传入 optimizer（建议在 Client 里持久化）
    - 返回 (avg_loss, n_tokens)
    """
    model.train()
    # 简单玩具数据；可替换为你的真实 batch
    texts = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "Question: 2+2=? Answer: 4.",
        "Question: Capital of France? Answer: Paris."
    ]

    # 有的 tokenizer 没有 pad_token，设为 eos 以避免警告/错误
    if tokenizer.pad_token_id is None and hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    total_loss = 0.0
    total_tokens = 0

    for _ in range(steps):
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        labels = input_ids.clone()
        # ✅ 屏蔽 PAD
        labels[attention_mask == 0] = -100

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()

        # ✅ 梯度裁剪，LoRA 小模型常用 0.5~1.0
        clip_grad_norm_(model.parameters(), 1.0)

        if optimizer is None:
            # 仍支持“手搓优化器”，但更安全
            for p in model.parameters():
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-lr)
                    p.grad.zero_()
        else:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_tokens += int(attention_mask.sum().item())

    avg_loss = total_loss / max(steps, 1)
    print(f"Local training loss: {avg_loss:.4f}, tokens={total_tokens}")
    return avg_loss, total_tokens



# from torch.utils.data import DataLoader
# from transformers import default_data_collator

# def train_one_step(model, tokenizer, device):

#     # 简单构造点假数据
#     texts = ["Hello, how are you?", "The quick brown fox jumps over the lazy dog."]
#     encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
#     input_ids = encodings["input_ids"].to(device)
#     attention_mask = encodings["attention_mask"].to(device)

#     labels = input_ids.clone()
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#     loss = outputs.loss

#     print(f"Local training loss: {loss.item():.4f}")
#     loss.backward()
#     for param in model.parameters():
#         if param.grad is not None:
#             param.data -= 1e-4 * param.grad  # 简单手动优化器
#             param.grad.zero_()
