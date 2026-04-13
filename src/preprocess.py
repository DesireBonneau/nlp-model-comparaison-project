from datasets import load_dataset
from sklearn.model_selection import train_test_split

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

# Second split: 50/50 of the 20% → 10% val, 10% test
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
# Expected: Train: 40000, Val: 5000, Test: 5000