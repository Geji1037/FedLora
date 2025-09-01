
from torch.utils.data import DataLoader
from transformers import default_data_collator

def train_one_step(model, tokenizer, device):

    # 简单构造点假数据
    texts = ["Hello, how are you?", "The quick brown fox jumps over the lazy dog."]
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    print(f"Local training loss: {loss.item():.4f}")
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            param.data -= 1e-4 * param.grad  # 简单手动优化器
            param.grad.zero_()
