- [Loss](#loss)
  - [Classification losses](#classification-losses)
    - [Cross entropy loss](#cross-entropy-loss)
    - [Binary Cross entropy loss](#binary-cross-entropy-loss)
  - [Regression Losses](#regression-losses)
    - [Mean Square Error Loss](#mean-square-error-loss)
    - [Huber loss](#huber-loss)
  - [Ranking losses](#ranking-losses)
    - [1. **Hinge Loss (Pairwise Ranking Loss)**](#1-hinge-loss-pairwise-ranking-loss)
    - [2. **Triplet Loss**](#2-triplet-loss)
    - [3. **Contrastive Loss**](#3-contrastive-loss)
    - [4. **RankNet Loss**](#4-ranknet-loss)
    - [5. **ListNet Loss**](#5-listnet-loss)
    - [6. **ListMLE (Listwise Learning to Rank Loss)**](#6-listmle-listwise-learning-to-rank-loss)
    - [7. **Mean Reciprocal Rank (MRR) Loss**](#7-mean-reciprocal-rank-mrr-loss)
    - [Summary](#summary)


# Loss

##	Classification losses

### Cross entropy loss 

交叉熵loss函数及原理

### Binary Cross entropy loss 



##	Regression Losses
### Mean Square Error Loss
### Huber loss



##	Ranking losses

https://gombru.github.io/2019/04/03/ranking_loss/

the objective of Ranking Losses is to predict relative distances between inputs. This task if often called metric learning

Here’s an organized breakdown of various rank loss functions, including their intuition, formula, advantages, and use cases:

### 1. **Hinge Loss (Pairwise Ranking Loss)**

**Intuition:**
Hinge loss aims to ensure that the relevant items are ranked higher than irrelevant items by a margin. It penalizes the model if the difference between the scores of a relevant and an irrelevant item is less than a certain threshold.

The intuition behind hinge loss, especially in the context of classification and ranking tasks, revolves around the concept of maximizing the margin between correctly and incorrectly classified instances or between relevant and irrelevant items in ranking. Here’s a breakdown of the key ideas:

- 1. **Margin Maximization**

In hinge loss, the goal is to not only classify or rank items correctly but to do so with a certain confidence margin. This margin is crucial for making the model more robust to noise and minor variations in the input data.

- 2. **Penalizing Misclassifications**

Hinge loss imposes a penalty when the predictions are not confident enough, i.e., when the decision boundary is too close to the instances. This penalty is zero when the instances are correctly classified with a sufficient margin, and it increases linearly as the margin decreases.

- 3. **Mathematical Formulation and Intuition**

For binary classification in Support Vector Machines (SVMs), the hinge loss for an instance \( (x_i, y_i) \) (where \( x_i \) is the input and \( y_i \) is the true label, either +1 or -1) is given by:

\[ L_{\text{hinge}}(x_i, y_i) = \max(0, 1 - y_i f(x_i)) \]

Here, \( f(x_i) \) is the prediction score or decision function.

- **Correct Classification with Margin:** If \( y_i f(x_i) \geq 1 \), the loss is zero. This means the instance is correctly classified with a margin of at least 1.
- **Incorrect Classification or Insufficient Margin:** If \( y_i f(x_i) < 1 \), the loss is positive, indicating either a misclassification or a correct classification with insufficient margin.

For ranking tasks, the hinge loss can be adapted to ensure that the score of a relevant item is higher than that of an irrelevant item by a margin:

\[ L_{\text{hinge}}(x_i, x_j) = \max(0, 1 - (f(x_i) - f(x_j))) \]

Here, \( x_i \) is a relevant item, \( x_j \) is an irrelevant item, and \( f(x) \) is the scoring function.

- **Correct Ranking with Margin:** If \( f(x_i) - f(x_j) \geq 1 \), the loss is zero. This means the relevant item \( x_i \) is ranked higher than the irrelevant item \( x_j \) with a margin of at least 1.
- **Incorrect Ranking or Insufficient Margin:** If \( f(x_i) - f(x_j) < 1 \), the loss is positive, indicating either an incorrect ranking or a correct ranking with insufficient margin.

- 4. **Geometric Interpretation**

In the geometric interpretation, hinge loss aims to find a decision boundary (hyperplane) that maximizes the distance (margin) between the boundary and the nearest data points from each class. This is fundamental to the concept of Support Vector Machines (SVMs).

- **Support Vectors:** The data points that lie closest to the decision boundary and have a direct impact on its position are called support vectors. Hinge loss is particularly sensitive to these points.
- **Robustness:** By maximizing the margin, hinge loss ensures that small perturbations in the input data do not easily cause misclassifications, leading to a more robust model.

- 5. **Advantages of Hinge Loss**

- **Robustness to Outliers:** By focusing on the margin, hinge loss helps in creating models that are less sensitive to outliers.
- **Sparse Solutions:** In the case of SVMs, hinge loss often leads to sparse solutions where only a subset of training data (support vectors) influences the decision boundary.
- **Generalization:** Models trained with hinge loss often have better generalization performance because the margin maximization helps prevent overfitting.


The intuition behind hinge loss is to ensure that not only are instances classified correctly or items ranked correctly, but they are done so with a confidence margin. This margin maximization leads to more robust and generalizable models by penalizing instances that are close to the decision boundary or are misclassified.

**Formula:**
\[ L_{\text{hinge}}(x_i, x_j) = \max(0, 1 - (f(x_i) - f(x_j))) \]
where \(x_i\) is a relevant item, \(x_j\) is an irrelevant item, and \(f(x)\) is the scoring function.

**Advantages:**
- **Simplicity:** Easy to implement and understand.
- **Effective for Binary Classification:** Works well in distinguishing relevant and irrelevant items.
- **Margin-based:** Creates a buffer zone, enhancing robustness to noise.

**Use Cases:**
- Document retrieval where the goal is to rank relevant documents higher than irrelevant ones.
- Ranking images based on relevance to a given query.



### 2. **Triplet Loss**

**Intuition:**
Triplet loss ensures that the distance between a positive pair (anchor and positive example) is smaller than the distance between a negative pair (anchor and negative example) by a margin.

**Formula:**
\[ L_{\text{triplet}}(a, p, n) = \max(0, d(a, p) - d(a, n) + \alpha) \]
where \(a\) is the anchor, \(p\) is a positive example, \(n\) is a negative example, \(d\) is the distance metric (e.g., Euclidean distance), and \(\alpha\) is the margin.

**Advantages:**
- **Direct Control Over Relative Distances:** Optimizes the relative distances in the embedding space.
- **Flexibility:** Suitable for various applications such as face recognition and image retrieval.
- **Effective for Fine-Grained Ranking:** Ensures relevant items are closer to the query than irrelevant items.

**Use Cases:**
- Face recognition systems.
- Image retrieval tasks.
- Recommender systems for personalized content ranking.

### 3. **Contrastive Loss**

**Intuition:**
Contrastive loss aims to bring similar items closer together and push dissimilar items further apart in the embedding space.

**Formula:**
\[ L_{\text{contrastive}}(x_i, x_j, y) = (1 - y) \cdot \frac{1}{2} \left(D_W^2\right) + y \cdot \frac{1}{2} \left(\max(0, m - D_W)\right)^2 \]
where \(D_W\) is the distance between the embeddings of \(x_i\) and \(x_j\), \(y\) is a binary label indicating whether \(x_i\) and \(x_j\) are similar or dissimilar, and \(m\) is the margin.

**Advantages:**
- **Simplicity:** Straightforward to implement.
- **Pairwise Distance Optimization:** Ensures similar items are close and dissimilar items are far apart.
- **Effective in Low-Dimensional Embedding Spaces:** Suitable for learning compact representations.

**Use Cases:**
- Learning embeddings for image similarity.
- Text similarity tasks.
- Signature verification systems.

### 4. **RankNet Loss**

**Intuition:**
RankNet uses a probabilistic framework to model the likelihood that one item is ranked higher than another, optimizing the relative ordering of items.

**Formula:**
\[ P(i > j) = \frac{1}{1 + e^{-(s_i - s_j)}} \]
\[ L_{\text{RankNet}} = -\log(P(i > j)) \]
where \(s_i\) and \(s_j\) are the scores for items \(i\) and \(j\).

**Advantages:**
- **Probabilistic Framework:** Provides intuitive and interpretable results.
- **Neural Network Based:** Easily integrates with deep learning models.
- **Pairwise Approach:** Effective for tasks focusing on relative ranking.

**Use Cases:**
- Web search ranking.
- Product recommendation systems.
- Document retrieval.

### 5. **ListNet Loss**

**Intuition:**
ListNet models the probability distribution of permutations, considering the entire list of items for better overall ranking performance.

**Formula:**
\[ P(\pi | s) = \prod_{i=1}^{n} \frac{e^{s_{\pi(i)}}}{\sum_{k=i}^{n} e^{s_{\pi(k)}}} \]
\[ L_{\text{ListNet}} = -\sum_{\pi} P(\pi | y) \log P(\pi | s) \]
where \(s\) is the score list, \(y\) is the ground-truth list, and \(\pi\) is a permutation.

**Advantages:**
- **Listwise Approach:** Considers the entire list of items.
- **Permutations Consideration:** Captures complex ranking relationships.
- **Global Optimization:** Focuses on the global ranking of items.

**Use Cases:**
- Learning to rank for search engines.
- Recommendation systems where the order of items is crucial.
- Sorting tasks in NLP applications.

### 6. **ListMLE (Listwise Learning to Rank Loss)**

**Intuition:**
ListMLE uses the likelihood of the correct permutation of items to optimize the entire ranking list, ensuring that the predicted ranking is as close as possible to the ground truth.

**Formula:**
\[ P(y | s) = \prod_{i=1}^{n} \frac{e^{s_{y_i}}}{\sum_{k=i}^{n} e^{s_{y_k}}} \]
\[ L_{\text{ListMLE}} = -\log P(y | s) \]
where \(y\) is the ground-truth permutation, and \(s\) is the predicted score list.

**Advantages:**
- **Likelihood-based:** Statistically sound approach.
- **End-to-End Optimization:** Optimizes the entire ranking list.
- **Permutation Sensitivity:** Ensures the ranking order is as close as possible to the ground truth.

**Use Cases:**
- Web search result ranking.
- Recommender systems optimizing for user engagement.
- Academic paper citation ranking.

### 7. **Mean Reciprocal Rank (MRR) Loss**

**Intuition:**
MRR evaluates the quality of the ranked list by focusing on the position of the first relevant item, emphasizing the importance of retrieving the most relevant item as early as possible.

**Formula:**
\[ MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i} \]
where \(\text{rank}_i\) is the rank position of the first relevant item for query \(i\).

**Advantages:**
- **Focus on Top Results:** Emphasizes the position of the first relevant item.
- **Interpretability:** Easy to interpret and understand.
- **Effective for Single Relevant Result:** Useful when each query typically has a single relevant result.

**Use Cases:**
- Search engine ranking evaluation.
- Question answering systems.
- Recommendation systems where the top result is crucial.

### Summary

These rank loss functions cater to different ranking tasks and objectives, each with unique advantages and suitable use cases. Choosing the appropriate rank loss function depends on the specific requirements of the application, such as whether the focus is on pairwise comparisons, listwise ranking, or prioritizing top results.
