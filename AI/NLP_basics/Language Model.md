# Language Model

模型：ULMFit , GPT, BERT，XLNET，Transformer XL

## BERT & ALBERT & GPT

Certainly! Here is the updated table including the parameter information for GPT-1:

| Model             | Parameters | Layers | Hidden Size | Self-Attention Heads |
|-------------------|------------|--------|-------------|----------------------|
| **BERT Small** (community variants) | ~30 million | 4-6    | 256-512     | 4-8                  |
| **BERT Base**     | 110 million | 12     | 768         | 12                   |
| **BERT Large**    | 340 million | 24     | 1024        | 16                   |
| **ALBERT Tiny**   | 5 million   | 4      | 312         | 12                   |
| **ALBERT Base**   | 12 million  | 12     | 768         | 12                   |
| **ALBERT Large**  | 18 million  | 24     | 1024        | 16                   |
| **ALBERT X-Large**| 60 million  | 24     | 2048        | 16                   |
| **ALBERT XX-Large**| 235 million | 12     | 4096        | 64                   |
| **GPT-1**         | 117 million | 12     | 768         | 12                   |
| **GPT-2 Small**   | 117 million | 12     | 768         | 12                   |
| **GPT-2 Medium**  | 345 million | 24     | 1024        | 16                   |
| **GPT-2 Large**   | 774 million | 36     | 1280        | 20                   |
| **GPT-2 XL**      | 1.5 billion | 48     | 1600        | 25                   |
| **GPT-3 Small**   | 125 million | 12     | 768         | 12                   |
| **GPT-3 Medium**  | 350 million | 24     | 1024        | 16                   |
| **GPT-3 Large**   | 760 million | 24     | 1536        | 16                   |
| **GPT-3 XL**      | 1.3 billion | 24     | 2048        | 16                   |
| **GPT-3 2.7B**    | 2.7 billion | 32     | 2560        | 32                   |
| **GPT-3 6.7B**    | 6.7 billion | 32     | 4096        | 32                   |
| **GPT-3 13B**     | 13 billion  | 40     | 5120        | 40                   |
| **GPT-3 175B**    | 175 billion | 96     | 12288       | 96                   |

This table now includes the parameter details for GPT-1, providing a more complete comparison of BERT, ALBERT, and GPT models.



Absolutely. This is actually a **very good interview question for an NLP/GenAI engineer**, because it starts with a simple parameter-count calculation and can naturally probe whether the candidate understands **Transformer architecture, embeddings, attention, memory footprint, and inference/training cost**.

One correction from our earlier table first: **BERT Base has about 110M parameters**, with:

* 12 Transformer encoder layers
* hidden size (H=768)
* 12 attention heads
* vocabulary size ≈ 30,522
* intermediate FFN size = 3,072
* maximum position embeddings = 512

I would structure the interview as a **progressive funnel**: start with a calculation, then progressively remove assumptions.

---

# 1. Main interview question

### Question

> **BERT Base has approximately 110 million parameters. Without looking up the answer, can you estimate the number of parameters from its architecture?**
>
> Assume:
>
> * Vocabulary size = 30,522
> * Hidden size = 768
> * Number of Transformer layers = 12
> * Number of attention heads = 12
> * Feed-forward intermediate size = 3,072
> * Maximum sequence length = 512
>
> Please break down your calculation into:
>
> 1. Token embeddings
> 2. Positional embeddings
> 3. One Transformer encoder layer
> 4. All 12 encoder layers
> 5. Pooler / other parameters
>
> Does your result approximately match the commonly quoted **110M parameters**?

This is much better than simply asking:

> "How many parameters does BERT Base have?"

because the latter tests memorization, while the former tests whether the candidate understands the architecture.

---

# 2. What a strong candidate should calculate

## A. Token embedding

BERT has:

[
V \times H
]

where:

* (V=30,522)
* (H=768)

Therefore:

[
30,522 \times 768
\approx 23.44M
]

So token embeddings contribute approximately:

**23.4M parameters**

---

## B. Position embeddings

Maximum sequence length is 512:

[
512 \times 768 = 393,216
]

Only about:

[
0.39M
]

So positional embeddings are relatively small.

---

## C. One self-attention block

The attention mechanism has four major matrices:

[
W_Q,W_K,W_V,W_O
]

Each is approximately:

[
768 \times 768
]

Therefore:

[
4 \times 768^2
]

[
=2,359,296
]

≈ **2.36M**

The important follow-up here is:

> **Why doesn't the number of attention heads change the parameter count?**

A strong candidate should explain that multi-head attention is usually implemented using combined projection matrices. Splitting 768 dimensions into 12 heads gives:

[
768/12=64
]

per head, but the total number of parameters remains approximately:

[
4H^2
]

---

# 3. Feed-forward network

This is where many candidates underestimate the parameter count.

BERT uses:

[
768 \rightarrow 3072 \rightarrow 768
]

So:

[
768\times3072 + 3072\times768
]

[
=2\times768\times3072
]

[
=4,718,592
]

≈ **4.72M**

So the FFN actually has roughly **twice as many parameters as the attention projections**.

This gives you a very nice interview follow-up:

> **Which contributes more parameters in a Transformer layer: self-attention or the FFN? Why?**

Expected answer:

> The FFN. For BERT Base, attention projections are about 2.36M while the FFN is about 4.72M.

---

# 4. LayerNorm and biases

There are two LayerNorms per Transformer block.

Each LayerNorm has:

* (\gamma): 768
* (\beta): 768

So:

[
2\times768\times2=3,072
]

Only a few thousand parameters.

The biases of the linear layers are also relatively small.

Therefore, a good approximation for one Transformer layer is:

[
2.36M + 4.72M \approx 7.08M
]

With 12 layers:

[
7.08M\times12
\approx84.9M
]

Then add embeddings:

[
84.9M + 23.4M + 0.4M
\approx108.7M
]

plus pooler, biases, etc.

Result:

[
\boxed{\approx110M}
]

That's the number you want the candidate to arrive at.

---

# 5. Recommended follow-up questions

I would then progressively increase the difficulty.

### Follow-up 1 — Architecture understanding

> **Why does BERT Base have 12 attention heads if the hidden size is only 768?**

Expected:

[
768/12=64
]

Each head operates on a 64-dimensional subspace.

---

### Follow-up 2 — Attention parameter count

> **If I increase the number of attention heads from 12 to 24 while keeping the hidden size at 768, will the number of parameters increase?**

Good answer:

**Approximately no.**

Because:

[
W_Q,W_K,W_V,W_O
]

are still approximately:

[
4\times768^2
]

The head dimension changes from 64 to 32.

This is a particularly good question because weaker candidates often assume:

> "More heads = more parameters."

---

### Follow-up 3 — Hidden size

> **What happens to the parameter count if we double the hidden size from 768 to 1536 while keeping the number of layers the same?**

This tests whether they understand the quadratic relationship.

Attention:

[
O(H^2)
]

FFN:

[
O(H\times4H)=O(H^2)
]

Embeddings:

[
O(VH)
]

Therefore, the Transformer parameters roughly grow **quadratically** with hidden size.

---

# 6. Sequence length — an excellent trap question

Ask:

> **If I increase BERT's maximum sequence length from 512 to 4,096, does the number of model parameters increase significantly?**

This is an excellent question.

For BERT-style learned positional embeddings:

[
512\times768 \rightarrow4096\times768
]

The positional embedding increases from:

[
0.39M
]

to:

[
3.15M
]

So the total model grows only by about **2.75M parameters**.

But computational cost is a completely different story.

Then immediately ask:

> **What happens to the computational complexity of self-attention?**

Expected:

[
O(n^2H)
]

where (n) is sequence length.

Going from 512 to 4096 means:

[
\left(\frac{4096}{512}\right)^2=64
]

So the attention matrix computation can become roughly **64× more expensive**.

This distinction is extremely valuable:

> **Parameter count ≠ computational cost.**

---

# 7. Memory follow-up

Then move from parameters to hardware.

> **If BERT Base has approximately 110M parameters, how much GPU memory is required just to store the parameters in FP32?**

Each FP32 parameter = 4 bytes.

Therefore:

[
110M\times4
\approx440MB
]

So roughly:

[
\boxed{440MB}
]

Then ask:

> **What about FP16?**

Approximately:

[
110M\times2
=220MB
]

Then:

> **Is 220 MB enough to run BERT Base inference?**

Expected answer:

**No.**

Because you also need:

* activations
* attention matrices
* temporary buffers
* CUDA/runtime overhead
* input/output tensors

---

# 8. Training-memory follow-up

Now you can make it much harder:

> **Suppose we fine-tune BERT Base using Adam in FP32. Is the model size still approximately 440 MB? What else do we need to store?**

A strong candidate should mention:

* model weights
* gradients
* Adam first moment (m)
* Adam second moment (v)
* activations

Very roughly, ignoring activations:

| Component    |         FP32 |
| ------------ | -----------: |
| Parameters   |      ~440 MB |
| Gradients    |      ~440 MB |
| Adam (m)     |      ~440 MB |
| Adam (v)     |      ~440 MB |
| **Subtotal** | **~1.76 GB** |

And real training memory is considerably higher because of **activations and temporary tensors**.

This makes a great transition from "parameter counting" to actual ML engineering.

---

# 9. A particularly good GenAI interview follow-up

Since you're interviewing **GenAI engineers**, I would also ask:

> **Suppose I want to fine-tune BERT Base using LoRA. Does LoRA reduce the 110M parameters of the original BERT model?**

The ideal answer:

> No. LoRA doesn't fundamentally reduce the base model's parameter count. The original parameters remain, but they are frozen. LoRA introduces a small number of trainable low-rank parameters.

Then ask:

> **If a linear layer is (768\times768), and LoRA rank (r=8), approximately how many trainable parameters does LoRA introduce?**

For one projection:

[
768\times8 + 8\times768
]

[
=12,288
]

versus:

[
768\times768=589,824
]

So LoRA reduces the **trainable parameter count** dramatically.

That's a much more meaningful question for a modern GenAI engineer than simply knowing "BERT has 110M parameters."

---

# 10. I would use this interview progression

You can essentially run the whole interview as one chain:

```text
BERT Base
   │
   ├── 1. Estimate total parameters
   │
   ├── 2. Calculate embedding parameters
   │
   ├── 3. Calculate attention parameters
   │
   ├── 4. Calculate FFN parameters
   │
   ├── 5. Why does FFN have more parameters?
   │
   ├── 6. Does number of heads affect parameter count?
   │
   ├── 7. What happens if hidden size doubles?
   │
   ├── 8. What happens if sequence length increases?
   │       ├── parameter count?
   │       └── computational complexity?
   │
   ├── 9. How much GPU memory for FP32?
   │
   ├── 10. How much memory for Adam training?
   │
   └── 11. How does LoRA change trainable parameters?
```

### Difficulty progression

| Question                 | What you're testing        |
| ------------------------ | -------------------------- |
| How many parameters?     | Basic knowledge            |
| Derive 110M              | Transformer architecture   |
| Calculate attention      | Linear algebra             |
| Calculate FFN            | Architecture understanding |
| Does head count matter?  | Multi-head attention       |
| Double hidden size       | Complexity reasoning       |
| Increase sequence length | Parameters vs compute      |
| FP32 memory              | Systems knowledge          |
| Adam memory              | Training mechanics         |
| LoRA                     | Modern GenAI engineering   |

**The key is not whether the candidate gets exactly 110M.** I'd care much more about whether they can construct the calculation from first principles and recognize the scaling laws:

[
\boxed{\text{Embedding}\sim VH}
]

[
\boxed{\text{Attention}\sim H^2}
]

[
\boxed{\text{FFN}\sim H^2}
]

[
\boxed{\text{Self-attention compute}\sim n^2H}
]

Those four relationships tell you a surprising amount about whether someone actually understands Transformer models rather than just memorizing model specifications.

