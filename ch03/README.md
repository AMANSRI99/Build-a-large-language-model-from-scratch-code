# Chapter 3: Coding attention mechanisms

Runnable code for chapter 3 of *Build a Large Language Model (From Scratch)* by Sebastian Raschka.
Each file is standalone and prints the same numbers as the book.

| File | Book section | What it builds |
| --- | --- | --- |
| [01_simple_self_attention.py](01_simple_self_attention.py) | 3.3 | Self-attention with no trainable weights: dot-product scores, softmax, weighted sum |
| [02_self_attention_trainable_weights.py](02_self_attention_trainable_weights.py) | 3.4 | Queries, keys and values; scaling by √d_k; `SelfAttention_v1` and `SelfAttention_v2` |
| [03_causal_attention.py](03_causal_attention.py) | 3.5 | Causal mask (tril and −∞ methods), dropout, `CausalAttention` |
| [04_multi_head_attention.py](04_multi_head_attention.py) | 3.6 | `MultiHeadAttentionWrapper` and the efficient `MultiHeadAttention` |

## Run

```bash
pip install torch
python ch03/01_simple_self_attention.py
```
