from preprocess import train_texts_clean, train_labels, val_texts_clean, val_labels, test_texts_clean, test_labels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import time

# Baseline Model 1: TF-IDF with Linear Classifier
# NOTE: BETTER VALUES WITH UNI+TRIGRAMS THAN WITH UNI+BIGRAMS

# Building the TF-IDF features (TF-IDF uses only the top 50,000 features from unigrams and bigrams)
# max_features is used to remove noise (words too rare to appear in both the training and testing sets)
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
# Fits only on the training data to avoid leakage
X_train_tfidf = tfidf.fit_transform(train_texts_clean)
# Gets the vectors for the validation and testing data
X_val_tfidf = tfidf.transform(val_texts_clean)
X_test_tfidf = tfidf.transform(test_texts_clean)

# Train Logistic Regression 
start = time.time()
lr_baseline = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
lr_baseline.fit(X_train_tfidf, train_labels)
# Recording the time that it took the model to learn the features
lr_train_time = time.time() - start

# Predict on the validation data
val_preds_lr = lr_baseline.predict(X_val_tfidf)
print(f"LR Baseline Accuracy: {accuracy_score(val_labels, val_preds_lr):.4f}")
print(f"Training time: {lr_train_time:.2f}s")
print(classification_report(val_labels, val_preds_lr))


# Baseline Model 2: Perceptron