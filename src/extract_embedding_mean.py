import os
os.environ["USE_TF"] = "0"
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_home/hub"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_home/transformers"

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from preprocess import train_texts, val_texts, test_texts

os.makedirs("/tmp/hf_home", exist_ok=True)
os.makedirs("embeddings_mean", exist_ok=True)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model.to(device)
model.eval()

def get_mean_pooled_embeddings(texts, batch_size=64, max_length=256):
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"].unsqueeze(-1)

            masked_embeddings = token_embeddings * attention_mask
            sum_embeddings = masked_embeddings.sum(dim=1)
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            mean_embeddings = sum_embeddings / lengths

            all_embeddings.append(mean_embeddings.cpu().numpy())
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_embeddings)

train_embeddings = get_mean_pooled_embeddings(train_texts, batch_size=64, max_length=256)
val_embeddings = get_mean_pooled_embeddings(val_texts, batch_size=64, max_length=256)
test_embeddings = get_mean_pooled_embeddings(test_texts, batch_size=64, max_length=256)

np.save("embeddings_mean/train_embeddings.npy", train_embeddings)
np.save("embeddings_mean/val_embeddings.npy", val_embeddings)
np.save("embeddings_mean/test_embeddings.npy", test_embeddings)

print("\nSaved mean-pooled embeddings:")
print("Train:", train_embeddings.shape)
print("Val:", val_embeddings.shape)
print("Test:", test_embeddings.shape)