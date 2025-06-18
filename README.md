# MNIST Classification with Custom SVD and Logistic Regression

This project implements Singular Value Decomposition (SVD) from scratch using NumPy to reduce the dimensionality of the MNIST dataset. A logistic regression classifier is trained on both the original and reduced datasets to analyze the trade-offs between dimensionality, accuracy, and training efficiency.

## Objective

- Apply SVD as a preprocessing step to reduce input dimensionality.
- Train a logistic regression classifier using the reduced features.
- Evaluate accuracy and training time across different SVD component counts.
- Visualize key results, including singular vectors as images.

## Learning Goals

- Understand SVD implementation and its impact on data representation.
- Explore dimensionality reduction techniques to improve model performance and efficiency.
- Compare trade-offs between accuracy and speed in machine learning pipelines.
- Practice performance evaluation through visual analytics.

---

## Dataset

- **MNIST Digits Dataset** (70,000 grayscale 28x28 images)
- Loaded via `fetch_openml` from `sklearn.datasets`.

---

## Methodology

### Preprocessing
- Normalize all pixel values to the range `[0, 1]`
- Split into training (80%) and testing (20%) sets

### SVD Implementation
- Custom NumPy-based implementation of SVD (no `scipy.linalg.svd` or `sklearn.decomposition`)
- Reduced dimensionality by selecting top `k` components
- Range tested: 1–220 components

### Classification
- Trained a multinomial logistic regression model (`solver='saga'`)
- Measured both training time and prediction accuracy for each `k`

---

## Results

### Accuracy and Training Time vs. SVD Components

| SVD Components | Accuracy | Training Time (s) |
|----------------|----------|-------------------|
| 1              | 0.1988   | 0.63              |
| 20             | 0.8779   | 1.74              |
| 60             | 0.9133   | 18.49             |
| 100            | 0.9181   | 60.35             |
| 140            | **0.9199** | 92.17           |
| 220            | 0.9209   | 160.55            |

**Insight**: Accuracy plateaus around 140 components, where the performance-time tradeoff is most favorable.

### Visualizations

- Accuracy increases with more components, plateauing ~140.
- Training time grows rapidly beyond 100+ components.
- Singular vectors (top 5) resemble distinct digit-like patterns when reshaped to 28x28.

