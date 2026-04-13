from datasets import load_dataset
from sklearn.model_selection import train_test_split
import re

# Load the IMDb film review dataset (originally has a 50/50 split for train/test) 
dataset = load_dataset("imdb")

# Combine all data 
all_texts = list(dataset['train']['text']) + list(dataset['test']['text'])
all_labels = list(dataset['train']['label']) + list(dataset['test']['label'])

# First split: 80% train, 20% temp
# random_state=42 is a seed number for reproducibility
# Stratifying on labels ensures the proportion of 0s and 1s in each resulting split matches the proportion in all_labels
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# Second split: 50/50 of the 20% -> 10% validation, 10% test
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# Expected: Train: 40000, Val: 5000, Test: 5000
print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)      # IMDb has HTML <br> tags
    text = re.sub(r'[^\w\s]', '', text)         # Remove punctuation
    text = text.lower().strip()
    return text


# Clean each text split
# NB: do not use those splits for the transformer model as BERT's tokenizer handles punctuation
train_texts_clean = [clean_text(t) for t in train_texts]
val_texts_clean = [clean_text(t) for t in val_texts]
test_texts_clean = [clean_text(t) for t in test_texts]