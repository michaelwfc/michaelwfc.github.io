
[sklearn algorithm cheat sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/)


# Feature selection

Feature selection is a crucial step in the machine learning workflow that involves selecting a subset of relevant features (variables, predictors) for model construction. By selecting the most important features, we can reduce the complexity of the model, improve performance, and reduce the risk of overfitting. Here are common methods for feature selection in machine learning:

## Types of Feature Selection Methods

### 1. **Filter Methods**:

  - **Univariate Selection**: Select features based on statistical tests (e.g., chi-square test, ANOVA, correlation coefficient).
  - **Variance Threshold**: Remove features with low variance.
  - **Correlation Matrix**: Select features based on their correlation with the target variable and each other.
  - **Mutual information**： Mutual Information measures the amount of information obtained about one random variable through another random variable. In the context of feature selection, it quantifies the amount of information gained about the target variable 𝑌 from knowing the feature 𝑋.


### 2. **Wrapper Methods**:
   - **Recursive Feature Elimination (RFE)**: Recursively remove least important features and build models on the remaining features.
   - **Forward Selection**: Start with no features and add one feature at a time that improves model performance.
   - **Backward Elimination**: Start with all features and remove one feature at a time that has the least impact on model performance.

### 3. **Embedded Methods**:
   - **Regularization Methods**: Methods like Lasso (L1 regularization) and Ridge (L2 regularization) penalize large coefficients, effectively performing feature selection.
   - **Tree-based Methods**: Decision trees, Random Forests, and Gradient Boosting Machines can be used to rank feature importance based on their contribution to model performance.

## Example Code for Feature Selection

Here are examples of how to perform feature selection using some of these methods in Python with `scikit-learn`:

#### Filter Method: Univariate Selection

```python
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, chi2

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Apply SelectKBest class to extract top 10 best features
best_features = SelectKBest(score_func=chi2, k=10)
fit = best_features.fit(X, y)

# Get the scores
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(data.feature_names)

# Concat two dataframes for better visualization
feature_scores = pd.concat([df_columns, df_scores], axis=1)
feature_scores.columns = ['Feature', 'Score']
print(feature_scores.nlargest(10, 'Score'))
```

#### Wrapper Method: Recursive Feature Elimination (RFE)

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Use Random Forest as the model
model = RandomForestClassifier()

# Recursive Feature Elimination
rfe = RFE(model, n_features_to_select=10)
fit = rfe.fit(X, y)

# Get selected features
selected_features = [data.feature_names[i] for i in range(len(rfe.support_)) if rfe.support_[i]]
print(selected_features)
```

#### Embedded Method: Lasso Regression

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LassoCV
import numpy as np

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Apply Lasso
lasso = LassoCV(cv=5).fit(X, y)

# Get the coefficients
importance = np.abs(lasso.coef_)
feature_names = data.feature_names

# Select features with non-zero coefficients
selected_features = feature_names[importance > 0]
print(selected_features)
```

#### Embedded Method: Tree-based Feature Importance

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_names = data.feature_names

# Create a DataFrame for better visualization
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances.head(10))
```

## Choosing the Right Method

The choice of feature selection method depends on several factors, including:

- **Dataset Size**: For large datasets, filter methods might be preferred due to their efficiency.
- **Model Type**: Some models, like tree-based methods, have built-in feature selection.
- **Interpretability**: Wrapper and embedded methods often provide more interpretable results.
- **Computational Resources**: Wrapper methods can be computationally expensive as they involve training multiple models.

## Summary

Feature selection is an essential step in building efficient and effective machine learning models. By reducing the number of features, we can simplify models, reduce overfitting, and improve performance. Various methods exist for feature selection, including filter, wrapper, and embedded methods, each with its advantages and suitable use cases. Selecting the appropriate method involves considering the specific characteristics of the dataset and the goals of the modeling task.


# Tuning Hyper Paramters

## Strategies:

- Manual Search: Adjust hyperparameters manually based on your understanding and intuition.
- Grid Search: Define a grid of hyperparameter values and evaluate all possible combinations.
- Random Search: Randomly sample hyperparameter values from predefined distributions.
- Automated Methods (e.g., Bayesian Optimization): Use algorithms that explore the hyperparameter space efficiently based on past evaluations.

Manual Search: Adjust hyperparameters manually based on your understanding and intuition.

## Grid Search: 

Define a grid of hyperparameter values and evaluate all possible combinations.

## Random Search: 

Randomly sample hyperparameter values from predefined distributions.

## Bayesian Optimization：

Use algorithms that explore the hyperparameter space efficiently based on past evaluations.



# Metrics

## Metrics for classification

A confusion matrix is a table that is used to evaluate the performance of a classification model. It summarizes the predictions made by a classifier compared to the actual labels of the data. It is a powerful tool for understanding the strengths and weaknesses of a model and is often used in various performance metrics calculation. 

### Components of a Confusion Matrix

A confusion matrix consists of four different combinations of predicted and actual labels:

1. **True Positives (TP)**:
   - These are the cases where the model predicted the positive class correctly.
   
2. **True Negatives (TN)**:
   - These are the cases where the model predicted the negative class correctly.
   
3. **False Positives (FP)**:
   - These are the cases where the model predicted the positive class incorrectly (it predicted positive, but the actual label was negative).
   
4. **False Negatives (FN)**:
   - These are the cases where the model predicted the negative class incorrectly (it predicted negative, but the actual label was positive).


|                   | Predicted:  (Positive) | Predicted:  (Negative) |
|-------------------|----------------------------|--------------------------------|
| Actual:       | True Positives (TP)        | False Negatives (FN)           |
| Actual:       | False Positives (FP)       | True Negatives (TN)            |

### Interpretation

- **True Positives (TP)**: Number of correctly predicted positive instances (spam emails correctly classified as spam).
- **True Negatives (TN)**: Number of correctly predicted negative instances (non-spam emails correctly classified as non-spam).
- **False Positives (FP)**: Number of incorrectly predicted positive instances (non-spam emails incorrectly classified as spam).
- **False Negatives (FN)**: Number of incorrectly predicted negative instances (spam emails incorrectly classified as non-spam).

### Usage of Confusion Matrix

The confusion matrix provides several metrics to evaluate the performance of a classification model:

- **Accuracy**: Overall accuracy of the model, calculated as \(\frac{TP + TN}{TP + TN + FP + FN}\).
- **Precision**: Proportion of positive predictions that were actually correct, calculated as \(\frac{TP}{TP + FP}\).
- **Recall (Sensitivity)**: Proportion of actual positives that were correctly predicted, calculated as \(\frac{TP}{TP + FN}\).
- **Specificity**: Proportion of actual negatives that were correctly predicted, calculated as \(\frac{TN}{TN + FP}\).
- **F1-score**: Harmonic mean of precision and recall, providing a balance between the two metrics.



## ROC & AUC

AUC (Area Under the Curve) and ROC (Receiver Operating Characteristic) curve are evaluation metrics commonly used in binary classification problems to assess the performance of machine learning models. Let's explore each of them in detail:

### ROC Curve (Receiver Operating Characteristic)

The ROC curve is a graphical plot that illustrates the performance of a binary classification model across various threshold settings. Here are the key components of the ROC curve:

- **True Positive Rate (Sensitivity/Recall)**: It is the proportion of actual positive cases correctly predicted by the model. Mathematically, it is defined as:
  \[
  \text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  \]
  where:
  - TP (True Positive): Number of correctly predicted positive instances.
  - FN (False Negative): Number of actual positive instances incorrectly predicted as negative.

- **False Positive Rate**: It is the proportion of actual negative cases incorrectly predicted as positive. It is defined as:
  \[
  \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
  \]
  where:
  - FP (False Positive): Number of actual negative instances incorrectly predicted as positive.
  - TN (True Negative): Number of correctly predicted negative instances.

The ROC curve is plotted by varying the threshold for predicting the positive class and calculating TPR and FPR at each threshold. A model with better classification accuracy will have an ROC curve that is closer to the top-left corner of the plot, indicating higher TPR and lower FPR across different thresholds.

### AUC (Area Under the Curve)

The AUC represents the area under the ROC curve. It quantifies the overall performance of a binary classification model across all possible thresholds. A higher AUC value (closer to 1) indicates better model performance in distinguishing between the positive and negative classes.

- **Interpretation of AUC**:
  - AUC = 1: Perfect classifier, which achieves a TPR of 1 and FPR of 0 across all thresholds.
  - AUC = 0.5: Classifier performs no better than random guessing (50-50 chance).
  - AUC < 0.5: Classifier performs worse than random guessing (inverse classification).




# Regulation

## LASSO(L1)

## Ridge(L2)
