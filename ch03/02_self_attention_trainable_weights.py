# Section 3.4: Implementing self-attention with trainable weights
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

# ---------------------------------------------------------------
# 3.4.1  Computing the attention weights step by step
# ---------------------------------------------------------------
x_2 = inputs[1]            # the second input element ("journey")
d_in = inputs.shape[1]     # the input embedding size, d_in=3
d_out = 2                  # the output embedding size, d_out=2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# query, key, value for "journey"
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("query_2:", query_2)

# keys and values for ALL inputs
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# Step 1: attention scores
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)        # omega_22
print("\nattn_score_22:", attn_score_22)

attn_scores_2 = query_2 @ keys.T           # all scores for query 2
print("attn_scores_2:", attn_scores_2)

# Step 2: scale by sqrt(d_k), then softmax
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("\nattn_weights_2:", attn_weights_2)

# Step 3: context vector = weighted sum of the value vectors
context_vec_2 = attn_weights_2 @ values
print("\ncontext_vec_2:", context_vec_2)

# ---------------------------------------------------------------
# 3.4.2  A compact self-attention class (listing 3.1)
# ---------------------------------------------------------------
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print("\nSelfAttention_v1 output:\n", sa_v1(inputs))

# ---------------------------------------------------------------
# Listing 3.2: using nn.Linear
# ---------------------------------------------------------------
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
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print("\nSelfAttention_v2 output:\n", sa_v2(inputs))

# ---------------------------------------------------------------
# Exercise 3.1: make v1 give the same output as v2
# (nn.Linear stores weights as (d_out, d_in), so transpose them)
# ---------------------------------------------------------------
sa_v1.W_query = nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key   = nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = nn.Parameter(sa_v2.W_value.weight.T)
print("\nExercise 3.1, v1 now equals v2:", torch.allclose(sa_v1(inputs), sa_v2(inputs)))
