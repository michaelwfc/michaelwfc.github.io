- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Leaning Rate Strategy](#leaning-rate-strategy)
  - [Warmup （bert）](#warmup-bert)
  - [Learning Rate Annealing：学习速率退火](#learning-rate-annealing学习速率退火)
  - [Step Decay](#step-decay)
  - [Exponential Decay](#exponential-decay)
  - [时序衰减学习率](#时序衰减学习率)
  - [余弦退火（Cosine annealing）](#余弦退火cosine-annealing)
  - [周期性学习率](#周期性学习率)
- [不均衡问题的处理](#不均衡问题的处理)
  - [Resampling Techniques](#resampling-techniques)
    - [a. **Oversampling the Minority Class:**](#a-oversampling-the-minority-class)
    - [b. **Undersampling the Majority Class:**](#b-undersampling-the-majority-class)
  - [Data Augmentation](#data-augmentation)
  - [Algorithmic Approaches](#algorithmic-approaches)
    - [a. **Cost-Sensitive Learning:**](#a-cost-sensitive-learning)
    - [b. **balance batch strategy**](#b-balance-batch-strategy)
    - [c. **Anomaly Detection:**](#c-anomaly-detection)
  - [Ensemble Methods](#ensemble-methods)
    - [a. **Balanced Random Forest:**](#a-balanced-random-forest)
    - [b. **EasyEnsemble and BalanceCascade:**](#b-easyensemble-and-balancecascade)
  - [Evaluation Metrics](#evaluation-metrics)
    - [a. **Use Appropriate Metrics:**](#a-use-appropriate-metrics)
  - [Specialized Models](#specialized-models)
    - [a. **XGBoost:**](#a-xgboost)
  - [Implementation Example: Using SMOTE and a Random Forest Classifier](#implementation-example-using-smote-and-a-random-forest-classifier)
    - [Conclusion](#conclusion)
- [Over-fitting \& Under-fitting](#over-fitting--under-fitting)
  - [High Bias/欠拟合 (train data performance)](#high-bias欠拟合-train-data-performance)
  - [High Variance/过拟合 (dev set performance)](#high-variance过拟合-dev-set-performance)
- [梯度累加](#梯度累加)

# Hyperparameter Tuning
 - Learning rate: $\alpha$
 - Optimizer:
   - Momentum: $\beta$
   - Aadm: $\beta_1, \beta_2$
 - mini batch size: $b$
 - hidden units size
 - learning rate decay
 - num of layers


# Leaning Rate Strategy

## Warmup （bert）

## Learning Rate Annealing：学习速率退火
先从一个比较高的学习速率开始然后慢慢地在训练中降低学习速率。这个方法背后的思想是我们喜欢快速地从初始参数移动到一个参数值「好」的范围，但这之后我们又想要一个学习速率小到我们可以发掘「损失函数上更深且窄的地方

[Karparthy 的 CS231n 课程笔](http://cs231n.github.io/neural-networks-3/#annealing-the-learning-rate)

## Step Decay
其中学习率经过一定数量的训练 epochs 后下降了一定的百分比。

## Exponential Decay

## 时序衰减学习率

## 余弦退火（Cosine annealing）
在采用批次随机梯度下降算法时，神经网络应该越来越接近Loss值的全局最小值。当它逐渐接近这个最小值时，学习率应该变得更小来使得模型不会超调且尽可能接近这一点。

余弦退火（Cosine annealing）利用余弦函数来降低学习率，进而解决这个问题，如下图所示：


## 周期性学习率
在上述论文中《Cyclical Learning Rates for Training Neural Networks》中，Leslie Smith 提出了一种周期性学习率表，可在两个约束值之间变动。如下图所示，它是一个三角形更新规则，但他也提到如何使用这一规则与固定周期衰减或指数周期衰减相结合。




# 不均衡问题的处理

Training a machine learning model with imbalanced data is a common challenge. Imbalanced data means that the classes in your dataset are not represented equally; one class might have significantly more examples than another. This imbalance can cause the model to perform poorly on the minority class. Here are several strategies to handle imbalanced data:

## Resampling Techniques

### a. **Oversampling the Minority Class:**
   - **Synthetic Minority Over-sampling Technique (SMOTE):** Generates synthetic samples for the minority class by interpolating between existing minority samples.
   - **Random Over-Sampling:** Randomly duplicates samples from the minority class.

### b. **Undersampling the Majority Class:**
   - **Random Under-Sampling:** Randomly removes samples from the majority class to balance the dataset.
   - **Cluster-based Under-Sampling:** Uses clustering to select representative samples from the majority class.

## Data Augmentation

   - Generate new data points by augmenting the minority class samples through transformations (like rotation, scaling, or noise addition).
  
## Algorithmic Approaches

### a. **Cost-Sensitive Learning:**
   - Modify the learning algorithm to penalize misclassification of the minority class more than the majority class. 
  
  For example, in decision trees, you can set class weights to give higher importance to the minority class.

f.nn.weighted_cross_entropy_with_logits
在损失函数中使用类权重。 本质上就是，让实例不足的类在损失函数中获得较高的权重，因此任何对该类的错分都将导致损失函数中非常高的错误。

带权重的 sigmoid 交叉熵 —— 适用于正、负样本数量差距过大时. 参数 pos_weight是Class Weight用来，对于 unbalanced数据非常有用。
增加了一个权重的系数，用来平衡正、负样本差距，可在一定程度上解决差距过大时训练结果严重偏向大样本的情况。

tf.losses.sigmoid_cross_entropy 和 tf.losses.softmax_cross_entropy 都支持权重Weight，
该 weight  为 Sample Weight 加上权重之后，在训练的时候能使某些sample比其他更加重要。
默认Weight为1，当w为标量的时候
tf.losses.sigmoid_cross_entropy(weight = w) = w* tf.losses.sigmoid_cross_entropy(weight = 1)
当W为向量的时候，权重加在每一个logits上再Reduce Mean.
f.losses.sigmoid_cross_entropy(weight = W) = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits*W)


###	b. **balance batch strategy** 
尽量让一个 batch 内，各类别的比例平衡


### c. **Anomaly Detection:**
   - Treat the minority class as anomalies and use anomaly detection techniques to identify them.

## Ensemble Methods

### a. **Balanced Random Forest:**
   - A variation of random forests where each tree is trained on a balanced bootstrap sample.

### b. **EasyEnsemble and BalanceCascade:**
   - Ensemble methods that focus on creating balanced datasets by either combining multiple undersampled datasets (EasyEnsemble) or iteratively undersampling the majority class (BalanceCascade).

## Evaluation Metrics

### a. **Use Appropriate Metrics:**
   - **Precision-Recall Curve:** More informative than ROC AUC when dealing with imbalanced datasets.
   - **F1 Score:** The harmonic mean of precision and recall, useful when you need a balance between precision and recall.
   - **Confusion Matrix:** Provides insights into true positives, false negatives, etc.
   - **Matthews Correlation Coefficient (MCC):** A balanced measure that can be used even if the classes are of very different sizes.



## Specialized Models

### a. **XGBoost:**
   - XGBoost and other boosting algorithms often have built-in parameters to handle imbalanced data, such as `scale_pos_weight` which can be set to the ratio of negative to positive examples.

## Implementation Example: Using SMOTE and a Random Forest Classifier

Here's a Python example using the `imbalanced-learn` library and `scikit-learn`:

```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assume X, y are your features and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

This code demonstrates the use of SMOTE for oversampling the minority class before training a Random Forest classifier. After training, it evaluates the model using a classification report which includes precision, recall, and F1 score.

### Conclusion

Handling imbalanced data requires careful consideration of resampling techniques, algorithmic modifications, appropriate evaluation metrics, and potentially data augmentation. By combining these strategies, you can train more effective models that perform well on both majority and minority classes.


# Over-fitting & Under-fitting

“欠拟合”常常在模型学习能力较弱，而数据复杂度较高的情况出现，此时模型由于学习能力不足，无法学习到数据集中的“一般规律”，因而导致泛化能力弱。
“过拟合”常常出现在模型学习能力过强的情况，此时的模型学习能力太强，以至于将训练集单个样本自身的特点都能捕捉到，并将其认为是“一般规律”，同样这种情况也会导致模型泛化能力下降。

过拟合与欠拟合的区别在于，欠拟合在训练集和测试集上的性能都较差，而过拟合往往能完美学习训练集数据的性质，而在测试集上的性能较差。

High Bias & low Variance
Low Bias & high Variance
High Bias & high Variance

![image](../images/Bias%20&%20Variance%20Tradeoff.png)

## High Bias/欠拟合 (train data performance)

当模型处于欠拟合状态时，根本的办法是增加模型复杂度。我们一般有以下一些办法：

	增加模型的迭代次数；
	更换描述能力更强的模型；
	生成更多特征供训练使用；如： Dense Feature + Sparse Feature
	降低正则化水平。

## High Variance/过拟合 (dev set performance)
当模型处于过拟合状态时，根本的办法是降低模型复杂度。我们则有以下一些武器：

	扩增训练集；
	减少训练使用的特征的数量；
	提高正则化水平。


# 梯度累加

变相扩大batch