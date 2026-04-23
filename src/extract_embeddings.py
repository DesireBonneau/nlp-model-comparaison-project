import os
os.environ["USE_TF"] = "0"

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from preprocess import train_texts, val_texts, test_texts

os.makedirs("embeddings", exist_ok=True)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


from tqdm import tqdm
import re

# Runs on GPU if available, otherwise uses CPU (slower)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Moves all of BERT's parameters onto selected device's memory
model.to(device)

# Implementing the batched embedding extraction
def extract_embeddings(texts, tokenizer, model, device, batch_size=32, max_length=512):
    """
    Extract [CLS] token embeddings from the last hidden state.
    Returns a numpy array of shape (len(texts), 768).
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize with padding and truncation
        encoded = tokenizer(
            batch_texts,
            # Pads the shorter texts in the batch to match the longest one (BERT needs uniform-length input)
            padding=True,
            truncation=True,
            max_length=max_length,
            # Returns PyTorch tensors
            return_tensors="pt"
        )
        # Moves the tokenized tensors to the same device as the model
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Forward pass so no gradients needed (runs the text through BERT layers)
        with torch.no_grad():
            outputs = model(**encoded)
        
        # Extract [CLS] token embedding (index 0) from last hidden state
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embeddings)
    
    # Each batch's embeddings are moved back to CPU as numpy arrays and stacked into one huge array
    return np.concatenate(all_embeddings, axis=0)


# For BERT, no need for full cleaning of the splits (only remove the <br> tags)
train_texts_bert = [re.sub(r'<br\s*/?>', ' ', t).strip() for t in train_texts]
val_texts_bert = [re.sub(r'<br\s*/?>', ' ', t).strip() for t in val_texts]
test_texts_bert = [re.sub(r'<br\s*/?>', ' ', t).strip() for t in test_texts]

print("Extracting train embeddings...")
X_train_emb = extract_embeddings(train_texts_bert, tokenizer, model, device)
print("Extracting val embeddings...")
X_val_emb = extract_embeddings(val_texts_bert, tokenizer, model, device)
print("Extracting test embeddings...")
X_test_emb = extract_embeddings(test_texts_bert, tokenizer, model, device)

# Making sure to save so that we don't have to recompute the embeddings
np.save("embeddings/train_embeddings.npy", X_train_emb)
np.save("embeddings/val_embeddings.npy", X_val_emb)
np.save("embeddings/test_embeddings.npy", X_test_emb)

print(f"Shapes: train={X_train_emb.shape}, val={X_val_emb.shape}, test={X_test_emb.shape}")
# Expected: (40000, 768), (5000, 768), (5000, 768)