- [Linear Regression](#linear-regression)
  - [Cost function： Mean Squared Error (MSE)](#cost-function-mean-squared-error-mse)
    - [Steps to Minimize the Cost Function](#steps-to-minimize-the-cost-function)
    - [Example Implementation](#example-implementation)
  - [Gradient Descent](#gradient-descent)
- [Logistic Regression](#logistic-regression)
  - [Cost function： binary cross-entropy](#cost-function-binary-cross-entropy)
- [Decision Tree](#decision-tree)
  - [Cost Function for Classification Decision Tree](#cost-function-for-classification-decision-tree)
    - [Gini Impurity Definition](#gini-impurity-definition)
    - [Calculation Example](#calculation-example)
    - [Interpretation](#interpretation)
    - [Splitting with Gini Impurity](#splitting-with-gini-impurity)
    - [Example: Decision Tree Split](#example-decision-tree-split)
    - [Summary](#summary)
- [Random Forest](#random-forest)
  - [Key Characteristics:](#key-characteristics)
- [GBM](#gbm)
  - [Algorithm](#algorithm)
  - [Key Characteristics:](#key-characteristics-1)
  - [How Gradient Boosting Machine (GBM) Works](#how-gradient-boosting-machine-gbm-works)
  - [Example Implementation](#example-implementation-1)
  - [Explanation of Hyperparameters](#explanation-of-hyperparameters)
  - [Differences between Random Forest and GBM:](#differences-between-random-forest-and-gbm)


# Linear Regression

For a linear regression model, where the hypothesis (predicted value) \( h_\theta(x) \) is defined as:

\[ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n \]

## Cost function： Mean Squared Error (MSE)
In linear regression, the cost function quantifies the error between the predicted values and the actual values. The most common cost function used for linear regression is the **Mean Squared Error (MSE)**. The MSE measures the average of the squares of the errors, that is, the difference between the actual values and the predicted values. 


The cost function \( J(\theta) \) is defined as:

\[ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2 \]

Where:
- \( m \) is the number of training examples.
- \( x^{(i)} \) is the feature vector of the \(i\)-th training example.
- \( y^{(i)} \) is the actual output value of the \(i\)-th training example.
- \( h_\theta(x^{(i)}) \) is the predicted value for the \(i\)-th training example.
- \( \theta \) is the vector of parameters (coefficients).

The factor \( \frac{1}{2m} \) is used for convenience in the derivation of the gradient descent algorithm, as it simplifies the derivative calculations.

### Steps to Minimize the Cost Function

To find the optimal parameters \( \theta \) that minimize the cost function, the gradient descent algorithm is typically used. The gradient descent algorithm iteratively updates the parameters as follows:

1. Initialize the parameters \( \theta \) (often with zeros or small random values).
2. Compute the predictions \( h_\theta(x) \) for all training examples.
3. Compute the cost function \( J(\theta) \).
4. Update the parameters using the gradient of the cost function:

\[ \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)} \]

Where:
- \( \alpha \) is the learning rate, a small positive value that controls the step size of each iteration.
- \( x_j^{(i)} \) is the \( j \)-th feature of the \( i \)-th training example.

### Example Implementation

Here's a simple implementation of linear regression using gradient descent in Python:

```python
import numpy as np

# Sample data
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.array([6, 8, 9, 11])

# Add a column of ones to X for the intercept term
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Initialize parameters
theta = np.zeros(X_b.shape[1])

# Learning rate
alpha = 0.01

# Number of iterations
n_iterations = 1000

# Number of training examples
m = X_b.shape[0]

# Gradient descent
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - alpha * gradients

print("Parameters:", theta)

# Predictions
predictions = X_b.dot(theta)
print("Predictions:", predictions)
```

This example demonstrates a basic implementation of linear regression using gradient descent. The `X` matrix is augmented with a column of ones to account for the intercept term \( \theta_0 \). The parameters are initialized and updated iteratively based on the gradients until convergence.

In summary, the cost function for a linear regression model is typically the Mean Squared Error (MSE), and minimizing this cost function through gradient descent or other optimization techniques allows us to find the best-fitting linear model for the data.

## Gradient Descent

# Logistic Regression

## Cost function： binary cross-entropy
- 


# Decision Tree

## Cost Function for Classification Decision Tree

For classification tasks, decision trees aim to partition the data in a way that maximizes the purity of the nodes. The two most common impurity measures used as cost functions are Gini impurity and Entropy (or Information Gain).

Gini impurity is a metric used in classification decision trees to measure the purity or impurity of a dataset. Understanding Gini impurity is crucial for comprehending how decision trees make decisions about splitting data. Here’s a detailed explanation:

### Gini Impurity Definition

Gini impurity quantifies the probability of incorrectly classifying a randomly chosen element from the dataset if it was randomly labeled according to the distribution of labels in the node. It ranges from 0 (pure node) to 0.5 (maximum impurity with binary classification).

The formula for Gini impurity for a node is:

\[ \text{Gini}(D) = 1 - \sum_{i=1}^{c} p_i^2 \]

Where:
- \( D \) is the dataset at the node.
- \( c \) is the number of classes.
- \( p_i \) is the proportion of instances of class \( i \) in the node.

### Calculation Example

Suppose you have a dataset with three classes (A, B, and C) at a particular node with the following distribution:
- Class A: 50%
- Class B: 30%
- Class C: 20%

To calculate the Gini impurity:

1. Compute the squared proportion of each class:
   - \( p_A^2 = (0.5)^2 = 0.25 \)
   - \( p_B^2 = (0.3)^2 = 0.09 \)
   - \( p_C^2 = (0.2)^2 = 0.04 \)

2. Sum these squared proportions:
   - \( \sum_{i=1}^{c} p_i^2 = 0.25 + 0.09 + 0.04 = 0.38 \)

3. Subtract this sum from 1 to get the Gini impurity:
   - \( \text{Gini}(D) = 1 - 0.38 = 0.62 \)

Thus, the Gini impurity for this node is 0.62.

### Interpretation

- **Gini Impurity of 0:** The node is pure, meaning all elements belong to a single class.
- **Higher Gini Impurity:** Indicates more mixed classes in the node, meaning the node is less pure.

### Splitting with Gini Impurity

When building a decision tree, the algorithm evaluates potential splits by calculating the Gini impurity for each possible split and choosing the one that results in the lowest Gini impurity for the child nodes. The goal is to split the dataset into parts where each part has a higher proportion of a single class, thereby reducing the overall Gini impurity.

### Example: Decision Tree Split

Consider a dataset with binary classes (0 and 1). Assume the parent node has:
- 10 samples of class 0
- 10 samples of class 1

The Gini impurity for the parent node is:

\[ \text{Gini}(\text{parent}) = 1 - (0.5^2 + 0.5^2) = 1 - 0.25 - 0.25 = 0.5 \]

Now, suppose a split results in two child nodes:
- Left child: 5 samples of class 0 and 5 samples of class 1
- Right child: 5 samples of class 0 and 5 samples of class 1

The Gini impurity for each child node is:

\[ \text{Gini}(\text{left child}) = 1 - (0.5^2 + 0.5^2) = 0.5 \]
\[ \text{Gini}(\text{right child}) = 1 - (0.5^2 + 0.5^2) = 0.5 \]

The weighted average Gini impurity for the split is:

\[ \text{Weighted Gini} = \frac{10}{20} \cdot 0.5 + \frac{10}{20} \cdot 0.5 = 0.5 \]

In this case, the split does not reduce the Gini impurity. The algorithm would search for a better split that lowers the impurity.

### Summary

Gini impurity is a measure used to evaluate the quality of a split in classification decision trees:
- **Lower Gini Impurity:** Indicates more homogenous nodes with predominantly one class.
- **Higher Gini Impurity:** Indicates more heterogeneous nodes with a mix of classes.

Decision trees aim to minimize Gini impurity at each split, thereby creating more pure nodes and improving the accuracy of the classification.

# Random Forest

Random Forest and Gradient Boosting Machine (GBM) are both ensemble learning methods that combine multiple decision trees to improve performance over a single tree. However, they have different approaches and characteristics. Here’s a detailed comparison:

## Key Characteristics:

- Bagging (Bootstrap Aggregating):

Random Forest uses bagging, where multiple decision trees are trained independently on different bootstrap samples of the dataset (random subsets with replacement).
The final prediction is made by averaging the predictions (for regression) or taking a majority vote (for classification) from all the trees.

- Feature Randomness:

During the training of each tree, a random subset of features is considered for splitting at each node, adding additional randomness and reducing correlation among trees.

- Parallel Training:

Trees in a Random Forest are trained independently and can be trained in parallel, making the process faster.




# GBM
Gradient Boosting Machine (GBM) is an ensemble learning method that builds models sequentially, where each new model tries to correct the errors made by the previous models. 
Key hyperparameters like the number of trees, learning rate, and tree depth need careful tuning to balance model performance and prevent overfitting. 


[XGboost](https://xgboost.readthedocs.io/en/stable/index.html)
[lightgbm]

## [Algorithm](https://en.wikipedia.org/wiki/Gradient_boosting#Algorithm)  


## Key Characteristics:

- Boosting:
GBM uses boosting, where trees are added sequentially, each new tree attempting to correct the errors of the combined ensemble of all previous trees.
Trees are built in a dependent manner, focusing on the residual errors of the existing model.
- Gradient Descent:
GBM minimizes the loss function by fitting new trees to the negative gradient of the loss function with respect to the current model’s predictions.
- Learning Rate:
A learning rate (shrinkage) parameter scales the contribution of each new tree, controlling the step size in the optimization process.

## How Gradient Boosting Machine (GBM) Works


Here's a step-by-step explanation of how it works, followed by an example with code and hyperparameters.

For a regression problem with a loss function \( L(y, F(x)) \), where \( y \) is the true value, and \( F(x) \) is the predicted value, the gradient boosting algorithm proceeds as follows:

1. Initialize the model with a constant value:
   \[
   F_0(x) = \arg\min_c \sum_{i=1}^{n} L(y_i, c)
   \]

2. For each iteration \( m = 1 \) to \( M \):

   - Compute the pseudo-residuals:
     \[
     r_{i}^{(m)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x) = F_{m-1}(x)}
     \]
      - For regression, th loss function is Mean Squred Error
      - For classification, the loss function is Binary Cross-entropy

   - Fit a base learner \( h_m(x) \) to the pseudo-residuals:
     \[
     h_m(x) \approx r_{i}^{(m)}
     \]

   - Compute multiplier \(\gamma_m\) by solving the following one-dimensional optimization problem
     \[
      \gamma_m =  \arg\min_\gamma \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) +\gamma h_m(x_i))
     \]  
   
   - Update the model:
     \[
     F_m(x) = F_{m-1}(x) + \gamma_m \cdot h_m(x)
     \]
     Where \( \nu \) is the learning rate.

## Example Implementation

Here’s an example of using GBM for a classification task using Python’s `scikit-learn` library:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the GBM model
gbm = GradientBoostingClassifier(
    n_estimators=100,      # Number of boosting stages (trees)
    learning_rate=0.1,     # Step size shrinkage
    max_depth=3,           # Maximum depth of each tree
    subsample=0.8,         # Fraction of samples used for fitting the individual trees
    max_features='sqrt',   # Number of features to consider when looking for the best split
    random_state=42        # Random seed for reproducibility
)

gbm.fit(X_train, y_train)

# Make predictions
y_pred = gbm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print the feature importances
importances = gbm.feature_importances_
print("Feature importances:", importances)
```

## Explanation of Hyperparameters

- **n_estimators**: The number of boosting stages (trees) to be added to the ensemble. More trees can lead to better performance but also increase the risk of overfitting.
  
- **learning_rate**: A small positive number that controls the contribution of each tree. Lower values require more trees but can result in a more robust model.
  
- **max_depth**: The maximum depth of each individual tree. Shallow trees are often preferred to prevent overfitting.
  
- **subsample**: The fraction of samples used to fit each tree. Setting this to less than 1.0 introduces randomness and helps prevent overfitting (similar to bagging).
  
- **max_features**: The number of features to consider when looking for the best split. Using a subset of features can help in reducing overfitting and improving generalization.
  
- **random_state**: A seed used by the random number generator for reproducibility of the results.










## Differences between Random Forest and GBM:

| Aspect                  | Random Forest                                                | Gradient Boosting Machine                                     |
|-------------------------|--------------------------------------------------------------|--------------------------------------------------------------|
| **Ensemble Method**     | Bagging (Bootstrap Aggregating)                              | Boosting                                                     |
| **Tree Independence**   | Trees are trained independently                              | Trees are trained sequentially, each correcting the previous |
| **Training Speed**      | Faster (parallel training)                                   | Slower (sequential training)                                 |
| **Overfitting Risk**    | Lower (due to averaging)                                     | Higher (requires careful tuning)                             |
| **Model Complexity**    | Simpler (fewer hyperparameters)                              | More complex (more hyperparameters)                          |
| **Interpretability**    | Less interpretable than single trees                         | Less interpretable, but feature importance can be derived    |
| **Prediction**          | Average/majority vote                                        | Weighted sum of trees                                        |
| **Use Case**            | Robust to noise, large datasets, high-dimensional data       | High accuracy, customizable loss functions                   |

Both Random Forest and GBM are powerful ensemble methods, but they are suited to different scenarios. Random Forest is generally easier to use and less prone to overfitting, making it a good default choice for many tasks. GBM, on the other hand, can provide better performance when properly tuned and is often used in competitions and applications requiring high accuracy.