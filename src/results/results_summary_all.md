| Model               | Feature_Type    | Pooling     |   Validation_Accuracy |   Test_Accuracy |   Training_Time_sec |
|:--------------------|:----------------|:------------|----------------------:|----------------:|--------------------:|
| LinearSVC           | TF-IDF          | ngram=(1,2) |                0.914  |          0.9158 |              0.7647 |
| LinearSVC           | TF-IDF          | ngram=(1,3) |                0.9104 |          0.9132 |              0.8623 |
| Logistic Regression | TF-IDF          | ngram=(1,2) |                0.91   |          0.909  |              3.5602 |
| Logistic Regression | TF-IDF          | ngram=(1,3) |                0.9114 |          0.9074 |              3.1408 |
| Perceptron          | TF-IDF          | ngram=(1,3) |                0.896  |          0.8954 |              0.2516 |
| Perceptron          | TF-IDF          | ngram=(1,2) |                0.8904 |          0.8896 |              0.315  |
| Logistic Regression | BERT Embeddings | CLS         |                0.8756 |          0.8808 |              7.0196 |
| LinearSVC           | BERT Embeddings | CLS         |                0.877  |          0.8802 |             12.561  |
| Logistic Regression | BERT Embeddings | Mean        |                0.8744 |          0.879  |              5.0136 |
| LinearSVC           | BERT Embeddings | Mean        |                0.8728 |          0.8768 |             11.9558 |
| Perceptron          | BERT Embeddings | Mean        |                0.7076 |          0.7116 |              0.5227 |
| Perceptron          | BERT Embeddings | CLS         |                0.6708 |          0.673  |              0.521  |