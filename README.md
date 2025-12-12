# Spanish Fake News Classifier: A Sequential Fine-Tuning Approach

### Introduction

The objective of this repository is to research and implement a robust classification system to distinguish between **real** and **fake** news articles written in Spanish.

The core architecture leverages a Transformer model—specifically `bert-base-spanish-wwm-uncased`—which has been pre-trained on a massive Spanish corpus. We employ a **Sequential Fine-Tuning** strategy to adapt this model to the specific domain of long-form news articles, overcoming data scarcity and length variation issues.

> **Important:**
> The actual model has not been uploaded to the repository as it's too heavy. You can find it in [HuggingFace](https://huggingface.co/Juanillaberia/spanish-fake-news-classifier).

### Data Collection

Focusing exclusively on Spanish news articles presented a significant challenge due to the scarcity of high-quality, consistent datasets. After an extensive search across platforms like Kaggle, HuggingFace, GitHub, and various international fact-checking sites, I selected four datasets that partially met the project requirements:

- **Spanish Fake and Real News** (Universidad de Madrid)
- **Spanish Political Fake News** (Universidad de Vigo)
- **Noticias falsas en español** (Kaggle - includes diverse sources)
- **Fake News Corpus Spanish** (by jpposadas - GitHub)

#### The Data Discrepancy Challenge

Upon analysis, a critical issue emerged: **Length Imbalance**.
Two of the datasets consisted of short text (approx. 50k samples), while the other two consisted of long-form articles (approx. 2k samples).

**Why we cannot simply mix them:**
Training a model on a combined dataset where one class (e.g., Fake) is predominantly short and the other (e.g., Real) is predominantly long introduces a **Length Bias**. The Transformer might learn to classify based on the _number of tokens_ rather than the _semantic content_ (e.g., "If it's short, it's fake"). Since our target production environment involves long articles, we need the model to learn structural and semantic patterns, not just text length.

### Methodology: Sequential Fine-Tuning

To solve the data imbalance and length discrepancy, we adopted a **Sequential Fine-Tuning** strategy. This process is divided into three distinct stages:

1. **Data Pre-Processing:** Cleaning, normalization, and tokenization.
2. **Stage 1 (Source Domain):** Pre-fine-tuning the `bert-base` model using the large, short-text dataset ($N \approx 50k$). The goal is to transfer general "fake news" linguistic patterns to the model.
3. **Stage 2 (Target Domain):** Fine-tuning the resulting model from Stage 1 using the smaller, long-text dataset ($N \approx 2k$). This adapts the model to the specific structure of long-form journalism.

---

### Data Pre-Processing

**1. Cleaning**
We first removed articles containing Null values:

- `50k dataset`: Removed ~2,000 Null values $\rightarrow$ **57,231** remaining articles.
- `2k dataset`: Contained 0 Null values $\rightarrow$ **2,141** remaining articles.

**2. Tokenization & Truncation**
We utilized the standard `HuggingFace Tokenizer` associated with the base model. Since BERT has a hard limit of `512 tokens`, we evaluated two truncation strategies for long articles:

- **Head Truncation:** Uses the first 512 tokens.
  - _Pros:_ Computationally efficient and preserves the introductory context where the main claim is usually stated.
  - _Cons:_ Loses the conclusion or later contradictions.
- **Head-Tail Truncation:** Uses the first _n_ tokens and the last _m_ tokens (concatenated).
  - _Pros:_ Captures both the introduction and the conclusion.
  - _Cons:_ Computationally more expensive to process sequentially and breaks the semantic flow of the text, potentially confusing the self-attention mechanism.

_Decision: We selected **Head Truncation** for computational efficiency and to maintain semantic continuity. Future research could explore Head-Tail or sliding window approaches._

**3. Splitting**

- **Stage 1 Dataset:** Train (80%) / Validation (20%).
- **Stage 2 Dataset:** Train (70%) / Validation (10%) / Testing (20%).

### Metrics

To ensure a comprehensive evaluation, we tracked four key metrics:

- **Accuracy:** Overall correctness.
- **Precision:** Usefulness for flagging fake news (minimizing false positives).
- **Recall:** Ability to find all fake news (minimizing false negatives).
- **F1-Score:** The harmonic mean of Precision and Recall.

---

### Stage 1: Pre-Fine-Tuning (Short Text)

We initialized `bert-base-spanish-wwm-uncased` with the following hyperparameters:

```python
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
num_train_epochs = 3
learning_rate = 2e-5
weight_decay = 0.01
logging_steps = 50
warmup_ratio = 0.1
save_total_limit = 2
```

**Freezing Strategy**: To prevent catastrophic forgetting of the pre-trained Spanish weights, we froze the majority of the model, unfreezing only:

1. The last 3 layers of the encoder.
2. The pooler layer.
3. The classifier layer.

| **Category** | **Parameters**  |
| ------------ | --------------- |
| Total        | **109,852,418** |
| Trainable    | 14,767,874      |
| Frozen       | 95,084,544      |

**Stage 1 Results**: Training was limited to 3 epochs to prevent overfitting on the short-text domain, as BERT models typically converge quickly.

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall   | F1       |
| ----- | ------------- | --------------- | -------- | --------- | -------- | -------- |
| 1     | 0.218800      | 0.193999        | 0.928802 | 0.931096  | 0.947487 | 0.939220 |
| 2     | 0.199600      | 0.176485        | 0.937364 | 0.949780  | 0.941920 | 0.945834 |
| 3     | 0.170900      | 0.168711        | 0.943479 | 0.944700  | 0.958772 | 0.951684 |

---

### Stage 2: Target Fine-Tuning (Long Text)

In this final stage, we loaded the weights from the `Stage 1 model`. We froze all layers except for the last 2 layers, the pooler and the classifier head to strictly adapt the decision boundary to the long-article structure without altering the learned linguistic features:

| **Category** | **Parameters**  |
| ------------ | --------------- |
| Total        | **109,852,418** |
| Trainable    | 7,680,002       |
| Frozen       | 102,172,416     |

**Hyperparameter Search**: We performed a grid search to optimize for the smaller dataset, using:

```python
per_device_train_batch_size = 16
per_device_eval_batch_size = 16
num_train_epochs = 5
learning_rate = 5e-6
weight_decay = 0.01
logging_steps = 10
warmup_ratio = 0.1
save_total_limit = 2
```

**Hyperparameter Search results:**

```python
{
 "learning_rate": 9.897764875650956e-05,
 "num_train_epochs": 4,
 "seed": 9,
 "per_device_train_batch_size": 32
}
```

With those hyperparameters we trained the model and got this final results:

| Epoch | Training Loss | Validation Loss | Accuracy | Precision | Recall   | F1       |
| ----- | ------------- | --------------- | -------- | --------- | -------- | -------- |
| 1     | 0.660900      | 0.544833        | 0.733645 | 0.818182  | 0.637168 | 0.716418 |
| 2     | 0.466700      | 0.455799        | 0.780374 | 0.761905  | 0.849558 | 0.803347 |
| 3     | 0.408400      | 0.437711        | 0.803738 | 0.770992  | 0.893805 | 0.827869 |
| 4     | 0.404900      | 0.425096        | 0.808411 | 0.776923  | 0.893805 | 0.831276 |

With the trained model we run predictions on the testing set and got this final results:

| Metric                  | Value   |
| ----------------------- | ------- |
| eval_loss               | 0.4183  |
| eval_accuracy           | 0.8205  |
| eval_precision          | 0.7835  |
| eval_recall             | 0.8702  |
| eval_f1                 | 0.8246  |
| eval_runtime (s)        | 14.0762 |
| eval_samples_per_second | 30.4770 |
| eval_steps_per_second   | 1.9180  |
| epoch                   | 4.0     |

---

### Model Analysis & Evaluation

After training, we conducted a comprehensive evaluation of the final model to understand its behavior beyond standard metrics. This analysis included:

- **Confusion Matrix Analysis:** Visual breakdown of true positives, false positives, true negatives, and false negatives to identify systematic classification patterns.
- **Confidence Distribution:** Examination of the model's prediction confidence across both classes to detect uncertainty patterns and potential calibration issues.
- **Error Analysis:** Deep dive into misclassified articles, with special focus on high-confidence errors (confidence > 0.8) that represent the most concerning failure mode in production.
- **Article Length Impact:** Investigation of whether model performance varies across different article lengths, ensuring the model generalizes well across the entire spectrum of long-form journalism.

These analyses revealed the model's strengths in detecting fake news patterns while highlighting specific edge cases that require attention. The evaluation framework established here enables continuous monitoring and iterative improvement of the classifier.

<br/>

# Conclusion

This project successfully demonstrates that **Sequential Fine-Tuning** is an effective strategy for building a Spanish fake news classifier capable of handling long-form articles despite significant data scarcity challenges.

### Key Achievements:

1. **Solved the Length Bias Problem:** By separating training into two stages, we prevented the model from learning spurious correlations based on article length rather than semantic content.

2. **Strong Performance on Long Articles:** The final model achieved **82.05% accuracy** and **82.46% F1-score** on the test set, demonstrating robust classification capability on complex, long-form journalism.

3. **Knowledge Transfer:** Stage 1 pre-fine-tuning on 57k short articles successfully transferred general "fake news" linguistic patterns, which were then adapted to long articles in Stage 2 with only ~2k samples.

4. **Comprehensive Evaluation:** Beyond standard metrics, our analysis revealed the model's confidence patterns, error modes, and performance consistency across article characteristics.

### Limitations & Future Work:

- **Dataset Size:** The long-article dataset remains relatively small (~2k samples). Future iterations could benefit from data augmentation or semi-supervised learning techniques.
- **Truncation Strategy:** Head truncation was chosen for efficiency, but exploring head-tail or sliding window approaches could capture more complete article context.
- **Domain Generalization:** Testing on diverse Spanish news sources and time periods would further validate the model's robustness.
- **Explainability:** Implementing attention visualization would provide deeper insights into which linguistic patterns drive classification decisions.

### Impact:

This classifier provides a foundation for automated fake news detection in Spanish-language media, a critical need given the prevalence of misinformation in Spanish-speaking communities. The sequential fine-tuning methodology can be adapted to other low-resource scenarios where data scarcity and distribution mismatch pose similar challenges.


