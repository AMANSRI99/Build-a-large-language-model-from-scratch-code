# Section 3.5: Hiding future words with causal attention
# Build a Large Language Model (From Scratch), Sebastian Raschka, chapter 3
import torch
import torch.nn as nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)
d_in = inputs.shape[1]   # 3
d_out = 2

# SelfAttention_v2 from section 3.4 (needed here: 3.5 reuses sa_v2)
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        return attn_weights @ values

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)

# ---------------------------------------------------------------
# 3.5.1  Applying a causal attention mask
# ---------------------------------------------------------------
# Ordinary attention weights first
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print("attn_weights:\n", attn_weights)

# Method 1: zero out above the diagonal, then renormalize
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("\nmask_simple:\n", mask_simple)

masked_simple = attn_weights * mask_simple
print("\nmasked_simple:\n", masked_simple)

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print("\nmasked_simple_norm:\n", masked_simple_norm)

# Method 2: fill future scores with -inf, then softmax (one step)
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("\nmasked:\n", masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print("\nattn_weights (method 2):\n", attn_weights)

# ---------------------------------------------------------------
# 3.5.2  Masking additional attention weights with dropout
# ---------------------------------------------------------------
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)    # dropout rate of 50%
example = torch.ones(6, 6)
print("\ndropout on ones:\n", dropout(example))

torch.manual_seed(123)
print("\ndropout on attn_weights:\n", dropout(attn_weights))

# ---------------------------------------------------------------
# 3.5.3  A compact causal attention class (listing 3.3)
# ---------------------------------------------------------------
batch = torch.stack((inputs, inputs), dim=0)
print("\nbatch.shape:", batch.shape)    # 2 inputs, 6 tokens, 3 numbers each

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("\ncontext_vecs:\n", context_vecs)
print("context_vecs.shape:", context_vecs.shape)
