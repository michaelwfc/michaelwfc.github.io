# Metircs
- **BLEU**: Focuses on precision and is commonly used for machine translation. It considers n-gram overlap and penalizes overly short translations.
- **Perplexity**: Measures how well a language model predicts the test data, focusing on the model's uncertainty. Lower perplexity indicates a better predictive model.
- **ROUGE**: Focuses on recall, measuring how much of the reference text is captured in the generated text. It is widely used in summarization and also applicable to machine translation.

Each metric serves a different purpose and is suitable for different types of NLP tasks. BLEU and ROUGE are more interpretable for tasks involving generated text comparison, while perplexity is a direct measure of a language model's quality.


## BLUE (Bilingual Evaluation Understudy)
**Purpose**: Commonly used for evaluating the quality of machine translation models.
**How it works**:
- BLEU measures how many words (or n-grams) in the generated text match the reference text.
- It uses precision to count the number of n-grams in the generated text that appear in the reference text.
- It applies a brevity penalty to penalize short translations that might match only a small part of the reference.

**Key Features**:
- **N-gram Matching**: Considers 1-gram (individual words), bigrams (pairs of words), trigrams, and up to n-grams.
- **Precision-Based**: Focuses on how much of the generated text overlaps with the reference.
- **Brevity Penalty**: Penalizes translations that are shorter than the reference to avoid short but incomplete translations.

**Use Case**: Primarily used in machine translation but can be applied to other text generation tasks like summarization and image captioning.

# Perplexity

**Purpose**: Used to evaluate language models, such as those used for text generation.
**How it works**:
- Perplexity measures how well a probability model predicts a sample.
- It is the exponentiation of the average negative log-likelihood of the test set.

**Key Features**:
- **Probability-Based**: Evaluates the likelihood of the model generating the test data.
- **Interpretability**: Lower perplexity indicates a better model. A perplexity of \( P \) means that on average, the model is as uncertain as if it had to choose between \( P \) options.
- **Log-Likelihood**: Directly related to the likelihood of the test data under the model.

**Use Case**: Commonly used in evaluating language models, such as those in automatic speech recognition and text generation.

# ROUGE(Recall-Oriented Understudy for Gisting Evaluation)

**Purpose**: Primarily used for evaluating automatic summarization and machine translation.
**How it works**:
- ROUGE measures the overlap between the generated text and the reference text.
- It focuses on recall, evaluating how much of the reference text appears in the generated text.

**Key Features**:
- **Recall-Based**: Measures the fraction of n-grams in the reference text that appear in the generated text.
- **Variants**:
  - **ROUGE-N**: N-gram recall.
  - **ROUGE-L**: Longest common subsequence (LCS) recall.
  - **ROUGE-W**: Weighted LCS.
  - **ROUGE-S**: Skip-bigram (pairs of words in their original order allowing for gaps).
- **Flexible Matching**: Can handle multiple reference summaries.

**Use Case**: Widely used in evaluating text summarization systems, but also applicable to machine translation and other text generation tasks.

