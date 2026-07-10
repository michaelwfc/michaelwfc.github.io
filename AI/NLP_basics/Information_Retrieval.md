
- [IR Pipeline](#ir-pipeline)
- [Indexing](#indexing)
- [Query Processing](#query-processing)
- [Retrieval Models](#retrieval-models)
- [Ranking](#ranking)
  - [TF-IDF](#tf-idf)
    - [**Components:**](#components)
  - [BM25 (Best Matching 25)](#bm25-best-matching-25)
    - [**Components:**](#components-1)
    - [Explanation of Term Frequency Saturation](#explanation-of-term-frequency-saturation)
    - [How BM25 Addresses Term Frequency Saturation](#how-bm25-addresses-term-frequency-saturation)
    - [Key Differences](#key-differences)
    - [Conclusion](#conclusion)
- [Evaluation](#evaluation)
  - [Evaluation Metrics](#evaluation-metrics)

# IR Pipeline

1. Indexing
2. Query Processing
3. Searching and Ranking:  
   - Retrieval Models: Applying retrieval models (e.g., Boolean, Vector Space, Probabilistic) to match the query with documents in the index.
   - Ranking Algorithms: Ordering the retrieved documents based on relevance scores. Common algorithms include TF-IDF (Term Frequency-Inverse Document Frequency) and BM25.
   - Relevance Feedback: Incorporating user feedback to refine and improve the search results.

# Indexing

# Query Processing

# Retrieval Models

# Ranking

## TF-IDF

Term Frequency-Inverse Document Frequency

TF-IDF (Term Frequency-Inverse Document Frequency) and BM25 (Best Matching 25) are both term weighting schemes used in information retrieval to evaluate the relevance of documents to a query. While they share some similarities, they also have key differences in their approach and implementation. Here's a detailed comparison:

**Overview:**

- TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection or corpus.
- It combines two measures: Term Frequency (TF) and Inverse Document Frequency (IDF).

###  **Components:**

1. **Term Frequency (TF)**: Measures how frequently a term occurs in a document.
   \[ \text{TF}(t, d) = \frac{f_{t,d}}{\sum_{k} f_{k,d}} \]
   where \( f_{t,d} \) is the number of times term \( t \) appears in document \( d \), and \( \sum_{k} f_{k,d} \) is the total number of terms in document \( d \).

2. **Inverse Document Frequency (IDF)**: Measures how important a term is across the entire corpus.
   \[ \text{IDF}(t) = \log \left( \frac{N}{n_t} \right) \]
   where \( N \) is the total number of documents, and \( n_t \) is the number of documents containing the term \( t \).

3. **TF-IDF Score**: The product of TF and IDF.
   \[ \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t) \]

**Characteristics:**

- Simple to compute and widely used in many text retrieval and text mining applications.
- Does not handle variations in document length effectively.
- Assumes term frequency is the primary indicator of importance, without considering term saturation or diminishing returns.

## BM25 (Best Matching 25)

- [BM25 实用详解 - 第 2 部分：BM25 算法及其变量](https://www.elastic.co/cn/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)

**Overview:**
    - BM25 is a probabilistic-based ranking function part of the Okapi BM25 family.
    - It is considered an improvement over traditional TF-IDF, addressing some of its limitations.

### **Components:**

1. **Term Frequency (TF)**: BM25 uses a non-linear term frequency component.
   \[ \text{TF}(t, d) = \frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})} \]

Where:

- \( f_{t,d} \) is the term frequency of term \( t \) in document \( d \).
- \( k_1 \) is a parameter that controls the degree of saturation (typically between 1.2 and 2.0).
- \( b \) is a parameter for document length normalization (typically around 0.75).
- \( |d| \) is the length of document \( d \) in words.
- \( \text{avgdl} \) is the average document length in the corpus.

2. **Inverse Document Frequency (IDF)**: BM25's IDF is similar to TF-IDF but often includes a slight modification.
   \[ \text{IDF}(t) = \log \left( \frac{N - n_t + 0.5}{n_t + 0.5} + 1 \right) \]
   where \( N \) is the total number of documents, and \( n_t \) is the number of documents containing the term \( t \).

3. **BM25 Score**: Combines the term frequency and IDF with adjustments for document length.
   \[ \text{BM25}(t, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})} \]

**Characteristics:**

- Adjusts term frequency saturation, meaning additional occurrences of a term have diminishing returns.
- Incorporates document length normalization to handle variations in document size more effectively.
- Parameters \( k_1 \) (typically between 1.2 and 2.0) and \( b \) (typically around 0.75) control term frequency saturation and document length normalization, respectively.


### Explanation of Term Frequency Saturation

Term frequency saturation refers to the phenomenon where the impact or importance of a term in a document does not increase linearly with its frequency. In simpler terms, as a term appears more frequently in a document, the additional significance of each new occurrence of the term diminishes.


Term frequency saturation is an important concept in information retrieval to prevent the overemphasis of highly frequent terms. Models like BM25 implement this by using non-linear functions that reduce the marginal gain of additional term occurrences, thereby improving the relevance and quality of search results.

1. **Linear Term Frequency (TF)**:
   - In the basic TF model, each occurrence of a term in a document is treated as equally important.
   - For instance, if a term appears 10 times in one document and 5 times in another, the term frequency would simply be 10 and 5, respectively.
   - This linear approach can lead to an overemphasis on terms that happen to appear very frequently, even if they are not as informative or significant after a certain point.

2. **Non-Linear Term Frequency (Saturation)**:
   - To address this, term frequency saturation introduces a non-linear relationship where the importance of each additional occurrence of a term decreases.
   - This concept is akin to diminishing returns: the first few occurrences of a term contribute significantly to its importance, but subsequent occurrences contribute less and less.
   - This is particularly useful for penalizing overly frequent terms that might not add much additional meaning or relevance after a certain frequency threshold.

### How BM25 Addresses Term Frequency Saturation

- **Control Parameter \( k_1 \)**: This parameter adjusts how quickly the saturation occurs. A higher \( k_1 \) value makes the saturation effect less pronounced, while a lower \( k_1 \) value increases the saturation effect.
- **Diminishing Returns**: As \( f_{t,d} \) (the term frequency) increases, the denominator grows faster, leading to a smaller increase in the overall TF score. This means that additional occurrences of the term have less impact on the final score.
- **Document Length Normalization**: The term frequency component also normalizes for document length, ensuring that longer documents do not get unfairly advantaged simply because they have more terms overall.



### Key Differences

1. **Term Frequency Handling**:
   - **TF-IDF**: Uses a linear term frequency which can overemphasize frequent terms.
   - **BM25**: Uses a non-linear term frequency, addressing term saturation and giving diminishing returns for repeated terms.

2. **Document Length Normalization**:
   - **TF-IDF**: Does not inherently normalize for document length, which can bias towards longer documents.
   - **BM25**: Explicitly normalizes for document length, making it more robust across documents of varying lengths.

3. **Parameter Tuning**:
   - **TF-IDF**: No parameters to tune.
   - **BM25**: Includes parameters \( k_1 \) and \( b \) that can be tuned to optimize performance for specific datasets.

4. **Complexity and Flexibility**:
   - **TF-IDF**: Simpler and easier to implement but less flexible in handling various document characteristics.
   - **BM25**: More complex but offers greater flexibility and usually better performance in modern IR systems.

### Conclusion

While both TF-IDF and BM25 are effective term weighting schemes, BM25 is generally considered more advanced and effective, especially in dealing with the nuances of term frequency saturation and document length normalization. It is widely used in modern search engines and information retrieval systems due to its robustness and flexibility.

# Evaluation

## Evaluation Metrics

- Mean Average Precision (MAP)
  The mean of the average precision scores for a set of queries.
