# Word Embedding Models

## CBOW

## FastText

## ELMo

## Word2Vec
Word2Vec (short for "Word to Vector") is a popular technique in natural language processing (NLP) for embedding words into vector space. Developed by Tomas Mikolov and his team at Google in 2013, it aims to capture the semantic meaning of words based on their context within a corpus of text. Here’s a detailed explanation:

### Concept
Word2Vec represents words as dense vectors in a continuous vector space. These vectors are learned in such a way that words with similar meanings are close to each other in this space. There are two main architectures used in Word2Vec:

1. **Continuous Bag of Words (CBOW)**: Predicts the target word (the word in the center) from the context words (surrounding words). The model takes the context words as input and tries to predict the target word.

2. **Skip-gram**: Predicts the context words from the target word. The model takes the target word as input and tries to predict the context words. This approach works well with small amounts of data and represents rare words or phrases better.

### Advantages
- **Semantic Similarity**: Words with similar meanings tend to have similar vectors, capturing semantic relationships.
- **Efficient Computation**: The models are relatively simple and computationally efficient compared to other deep learning models.
- **Transfer Learning**: Once trained, Word2Vec vectors can be used in various NLP tasks, often improving performance.


### Training

Training Word2Vec involves several key steps and concepts, including data preprocessing, model architecture, and optimization. Here’s a detailed breakdown of the training process:

#### 1. Data Preparation

Before training, the text data must be preprocessed:
    - **Tokenization**: Splitting the text into individual words.
    - **Vocabulary Building**: Creating a list of unique words (vocabulary) from the text.
    - **Context Window**: Defining the size of the context window, which determines how many words before and after the target word are considered as context.

#### 2. Model Architectures

Word2Vec uses two main architectures: Continuous Bag of Words (CBOW) and Skip-gram.

##### Continuous Bag of Words (CBOW)
- **Objective**: Predict the target word from the context words.
- **Input**: Context words (surrounding words).
- **Output**: Target word (center word).
- **Training Example**: For the sentence "the cat sits on the mat", with a window size of 2, for the target word "sits", the context words are ["the", "cat", "on", "the"].

##### Skip-gram
- **Objective**: Predict context words from the target word.
- **Input**: Target word (center word).
- **Output**: Context words (surrounding words).
- **Training Example**: Using the same sentence and window size, for the target word "sits", the context words are ["the", "cat", "on", "the"].

#### 3. Neural Network Structure
The neural network used in Word2Vec is shallow, consisting of an input layer, a hidden layer, and an output layer.

- **Input Layer**: One-hot encoded vector representing the input word.
- **Hidden Layer**: Dense layer with a lower dimensionality than the input. This layer holds the word embeddings.
- **Output Layer**: Produces a probability distribution over the vocabulary.

#### 4. Training Process
##### Initialization
- Initialize the weights of the hidden and output layers randomly or using a small constant.

##### Forward Pass
- **CBOW**: The input context words are averaged to produce the hidden layer representation, which is then used to predict the target word.
- **Skip-gram**: The input target word is used to predict each context word independently.

##### Softmax Function
- A softmax function is applied to the output layer to obtain a probability distribution over the vocabulary.

##### Loss Function
- **Negative Log-Likelihood**: The loss is typically computed using the negative log-likelihood of the predicted target word/context words.
- **Negative Sampling**: Instead of computing the softmax over the entire vocabulary (which is computationally expensive), negative sampling is often used. This involves only updating a small, random sample of negative examples (non-context words) and the positive examples (context words).

##### Backpropagation
- Compute gradients of the loss with respect to the weights.
- Update the weights using gradient descent or a variant like stochastic gradient descent (SGD).

#### 5. Optimization Techniques
- **Negative Sampling**: Reduces computational complexity by updating only a subset of weights.
- **Subsampling of Frequent Words**: Words that occur very frequently are downsampled to prevent them from dominating the training process.
- **Hierarchical Softmax**: An alternative to negative sampling that approximates the full softmax by using a binary tree representation of the output layer.


### Example with Skip-gram and Negative Sampling
Suppose we have the sentence "the quick brown fox jumps over the lazy dog" and we're training a Skip-gram model with a window size of 2:

1. **Target word**: "quick"
   - **Context words**: ["the", "brown"]
   - Generate training pairs: ("quick", "the") and ("quick", "brown")

2. **Forward Pass**:
   - Input: One-hot vector for "quick"
   - Hidden layer: Look up the embedding for "quick"
   - Output layer: Compute the probability distribution for the context words

3. **Negative Sampling**:
   - For each training pair, sample negative examples (words not in the context).

4. **Backward Pass and Update**:
   - Compute the gradients and update the weights using SGD.

By following these steps, the Word2Vec model learns word embeddings that capture semantic relationships based on the context in which words appear.
## GloVe