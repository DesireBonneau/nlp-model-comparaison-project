import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

from preprocess import (
    train_texts,
    val_texts,
    test_texts,
    train_texts_clean,
    val_texts_clean,
    test_texts_clean,
    train_labels,
    val_labels,
    test_labels,
)

os.makedirs("results", exist_ok=True)

results = []


def add_result(model_name, feature_type, pooling, val_acc, test_acc, train_time):
    results.append({
        "Model": model_name,
        "Feature_Type": feature_type,
        "Pooling": pooling,
        "Validation_Accuracy": round(float(val_acc), 4),
        "Test_Accuracy": round(float(test_acc), 4),
        "Training_Time_sec": round(float(train_time), 4),
    })


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, desc=""):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\n=== {desc} ===")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Training Time: {train_time:.2f}s")
    print("\n[Validation Report]")
    print(classification_report(y_val, val_preds))
    print("[Test Report]")
    print(classification_report(y_test, test_preds))

    return val_acc, test_acc, train_time, val_preds, test_preds


# --------------------------------------------------
# 1. TF-IDF (1,2) baselines
# --------------------------------------------------
print("Running TF-IDF (1,2) experiments...")

vectorizer_12 = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_12 = vectorizer_12.fit_transform(train_texts_clean)
X_val_12 = vectorizer_12.transform(val_texts_clean)
X_test_12 = vectorizer_12.transform(test_texts_clean)

# Logistic Regression
val_acc, test_acc, train_time, val_preds_lr_tfidf, test_preds_lr_tfidf = evaluate_model(
    LogisticRegression(max_iter=1000),
    X_train_12, X_val_12, X_test_12,
    train_labels, val_labels, test_labels,
    desc="TF-IDF (1,2) + Logistic Regression"
)
add_result("Logistic Regression", "TF-IDF", "ngram=(1,2)", val_acc, test_acc, train_time)

# Perceptron
val_acc, test_acc, train_time, _, _ = evaluate_model(
    Perceptron(max_iter=1000),
    X_train_12, X_val_12, X_test_12,
    train_labels, val_labels, test_labels,
    desc="TF-IDF (1,2) + Perceptron"
)
add_result("Perceptron", "TF-IDF", "ngram=(1,2)", val_acc, test_acc, train_time)

# LinearSVC
val_acc, test_acc, train_time, _, _ = evaluate_model(
    LinearSVC(),
    X_train_12, X_val_12, X_test_12,
    train_labels, val_labels, test_labels,
    desc="TF-IDF (1,2) + LinearSVC"
)
add_result("LinearSVC", "TF-IDF", "ngram=(1,2)", val_acc, test_acc, train_time)


# --------------------------------------------------
# 2. TF-IDF (1,3) baselines
# --------------------------------------------------
print("\nRunning TF-IDF (1,3) experiments...")

vectorizer_13 = TfidfVectorizer(max_features=50000, ngram_range=(1, 3))
X_train_13 = vectorizer_13.fit_transform(train_texts_clean)
X_val_13 = vectorizer_13.transform(val_texts_clean)
X_test_13 = vectorizer_13.transform(test_texts_clean)

# Logistic Regression
val_acc, test_acc, train_time, _, _ = evaluate_model(
    LogisticRegression(max_iter=1000),
    X_train_13, X_val_13, X_test_13,
    train_labels, val_labels, test_labels,
    desc="TF-IDF (1,3) + Logistic Regression"
)
add_result("Logistic Regression", "TF-IDF", "ngram=(1,3)", val_acc, test_acc, train_time)

# Perceptron
val_acc, test_acc, train_time, _, _ = evaluate_model(
    Perceptron(max_iter=1000),
    X_train_13, X_val_13, X_test_13,
    train_labels, val_labels, test_labels,
    desc="TF-IDF (1,3) + Perceptron"
)
add_result("Perceptron", "TF-IDF", "ngram=(1,3)", val_acc, test_acc, train_time)

# LinearSVC
val_acc, test_acc, train_time, _, _ = evaluate_model(
    LinearSVC(),
    X_train_13, X_val_13, X_test_13,
    train_labels, val_labels, test_labels,
    desc="TF-IDF (1,3) + LinearSVC"
)
add_result("LinearSVC", "TF-IDF", "ngram=(1,3)", val_acc, test_acc, train_time)


# --------------------------------------------------
# 3. CLS embedding models
# --------------------------------------------------
print("\nRunning CLS embedding experiments...")

train_cls = np.load("embeddings/train_embeddings.npy")
val_cls = np.load("embeddings/val_embeddings.npy")
test_cls = np.load("embeddings/test_embeddings.npy")

# Logistic Regression
val_acc, test_acc, train_time, val_preds_lr_cls, test_preds_lr_cls = evaluate_model(
    LogisticRegression(max_iter=1000),
    train_cls, val_cls, test_cls,
    train_labels, val_labels, test_labels,
    desc="BERT CLS + Logistic Regression"
)
add_result("Logistic Regression", "BERT Embeddings", "CLS", val_acc, test_acc, train_time)

# Perceptron
val_acc, test_acc, train_time, _, _ = evaluate_model(
    Perceptron(max_iter=1000),
    train_cls, val_cls, test_cls,
    train_labels, val_labels, test_labels,
    desc="BERT CLS + Perceptron"
)
add_result("Perceptron", "BERT Embeddings", "CLS", val_acc, test_acc, train_time)

# LinearSVC
val_acc, test_acc, train_time, _, _ = evaluate_model(
    LinearSVC(),
    train_cls, val_cls, test_cls,
    train_labels, val_labels, test_labels,
    desc="BERT CLS + LinearSVC"
)
add_result("LinearSVC", "BERT Embeddings", "CLS", val_acc, test_acc, train_time)


# --------------------------------------------------
# 4. Mean embedding models
# --------------------------------------------------
print("\nRunning mean embedding experiments...")

train_mean = np.load("embeddings_mean/train_embeddings.npy")
val_mean = np.load("embeddings_mean/val_embeddings.npy")
test_mean = np.load("embeddings_mean/test_embeddings.npy")

# Logistic Regression
val_acc, test_acc, train_time, _, _ = evaluate_model(
    LogisticRegression(max_iter=1000),
    train_mean, val_mean, test_mean,
    train_labels, val_labels, test_labels,
    desc="BERT Mean + Logistic Regression"
)
add_result("Logistic Regression", "BERT Embeddings", "Mean", val_acc, test_acc, train_time)

# Perceptron
val_acc, test_acc, train_time, _, _ = evaluate_model(
    Perceptron(max_iter=1000),
    train_mean, val_mean, test_mean,
    train_labels, val_labels, test_labels,
    desc="BERT Mean + Perceptron"
)
add_result("Perceptron", "BERT Embeddings", "Mean", val_acc, test_acc, train_time)

# LinearSVC
val_acc, test_acc, train_time, _, _ = evaluate_model(
    LinearSVC(),
    train_mean, val_mean, test_mean,
    train_labels, val_labels, test_labels,
    desc="BERT Mean + LinearSVC"
)
add_result("LinearSVC", "BERT Embeddings", "Mean", val_acc, test_acc, train_time)


# --------------------------------------------------
# 5. Save summary table
# --------------------------------------------------
df = pd.DataFrame(results)
df = df.sort_values(by=["Test_Accuracy", "Validation_Accuracy"], ascending=False)
df.to_csv("results/results_summary_all.csv", index=False)

print("\nFinal results summary:")
print(df)

# Also save markdown-friendly version
with open("results/results_summary_all.md", "w", encoding="utf-8") as f:
    f.write(df.to_markdown(index=False))


# --------------------------------------------------
# 6. Accuracy comparison graph
# --------------------------------------------------
labels = [
    f"{row['Model']}\n{row['Feature_Type']}\n{row['Pooling']}"
    for _, row in df.iterrows()
]
x = np.arange(len(df))
val_scores = df["Validation_Accuracy"].values
test_scores = df["Test_Accuracy"].values

plt.figure(figsize=(12, 7))
width = 0.35
plt.bar(x - width / 2, val_scores, width, label="Validation Accuracy")
plt.bar(x + width / 2, test_scores, width, label="Test Accuracy")
plt.xticks(x, labels, rotation=35, ha="right")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison Across All Experiments")
plt.legend()
plt.tight_layout()
plt.savefig("results/accuracy_comparison_all.png", dpi=300)
plt.close()


# --------------------------------------------------
# 7. Runtime comparison graph
# --------------------------------------------------
times = df["Training_Time_sec"].values

plt.figure(figsize=(12, 7))
plt.bar(x, times)
plt.xticks(x, labels, rotation=35, ha="right")
plt.ylabel("Training Time (seconds)")
plt.title("Training Time Comparison Across All Experiments")
plt.tight_layout()
plt.savefig("results/runtime_comparison_all.png", dpi=300)
plt.close()


# --------------------------------------------------
# 8. Error analysis:
#    TF-IDF (1,2) + LR  vs  BERT CLS + LR
# --------------------------------------------------
rows_tfidf_correct_cls_wrong = []
rows_cls_correct_tfidf_wrong = []

for i, (text, gold, p_tfidf, p_cls) in enumerate(
    zip(test_texts, test_labels, test_preds_lr_tfidf, test_preds_lr_cls)
):
    if p_tfidf == gold and p_cls != gold:
        rows_tfidf_correct_cls_wrong.append({
            "Index": i,
            "Gold_Label": gold,
            "TFIDF_LR_Pred": int(p_tfidf),
            "CLS_LR_Pred": int(p_cls),
            "Text": text,
        })
    elif p_cls == gold and p_tfidf != gold:
        rows_cls_correct_tfidf_wrong.append({
            "Index": i,
            "Gold_Label": gold,
            "TFIDF_LR_Pred": int(p_tfidf),
            "CLS_LR_Pred": int(p_cls),
            "Text": text,
        })

df_a = pd.DataFrame(rows_tfidf_correct_cls_wrong)
df_b = pd.DataFrame(rows_cls_correct_tfidf_wrong)

df_a.to_csv("results/tfidf12_lr_correct_cls_lr_wrong.csv", index=False)
df_b.to_csv("results/cls_lr_correct_tfidf12_lr_wrong.csv", index=False)

df_a.head(20).to_csv("results/tfidf12_lr_correct_cls_lr_wrong_sample.csv", index=False)
df_b.head(20).to_csv("results/cls_lr_correct_tfidf12_lr_wrong_sample.csv", index=False)

print("\nSaved files:")
print(" - results/results_summary_all.csv")
print(" - results/results_summary_all.md")
print(" - results/accuracy_comparison_all.png")
print(" - results/runtime_comparison_all.png")
print(" - results/tfidf12_lr_correct_cls_lr_wrong.csv")
print(" - results/cls_lr_correct_tfidf12_lr_wrong.csv")
print(" - results/tfidf12_lr_correct_cls_lr_wrong_sample.csv")
print(" - results/cls_lr_correct_tfidf12_lr_wrong_sample.csv")