# Section 3.3: A simple self-attention mechanism without trainable weights
# Build a Large Language Model (From Scratch), Sebastian Raschka, chapter 3
import torch

# ---------------------------------------------------------------
# 3.3.1  The input: 6 tokens, each embedded as 3 numbers
# ---------------------------------------------------------------
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],  # Your     (x^1)
     [0.55, 0.87, 0.66],  # journey  (x^2)
     [0.57, 0.85, 0.64],  # starts   (x^3)
     [0.22, 0.58, 0.33],  # with     (x^4)
     [0.77, 0.25, 0.10],  # one      (x^5)
     [0.05, 0.80, 0.55]]  # step     (x^6)
)

# ---------------------------------------------------------------
# Step 1: attention scores for ONE query ("journey")
# ---------------------------------------------------------------
query = inputs[1]                                   # 2nd input token is the query
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)        # dot product
print("Attention scores (journey):", attn_scores_2)

# What a dot product is: multiply element-wise, then sum
res = 0.
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
print("Manual dot product (Your . journey):", res)
print("torch.dot:", torch.dot(inputs[0], query))

# ---------------------------------------------------------------
# Step 2: normalize scores into weights that add up to 1
# ---------------------------------------------------------------
# (a) simple normalization: divide by the total
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("\nAttention weights (simple):", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# (b) naive softmax
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("\nAttention weights (naive softmax):", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# (c) PyTorch's softmax (numerically stable, use this one)
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("\nAttention weights (torch.softmax):", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# ---------------------------------------------------------------
# Step 3: context vector = weighted sum of all input vectors
# ---------------------------------------------------------------
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("\nContext vector z(2):", context_vec_2)

# ---------------------------------------------------------------
# 3.3.2  Attention weights for ALL inputs
# ---------------------------------------------------------------
# Step 1 for all pairs, with two loops (slow)
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print("\nAttention scores (loops):\n", attn_scores)

# Same result with one matrix multiplication (fast)
attn_scores = inputs @ inputs.T
print("\nAttention scores (inputs @ inputs.T):\n", attn_scores)

# Step 2 for all rows
attn_weights = torch.softmax(attn_scores, dim=-1)   # dim=-1: normalize each row
print("\nAttention weights:\n", attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("\nRow 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# Step 3 for all rows
all_context_vecs = attn_weights @ inputs
print("\nAll context vectors:\n", all_context_vecs)
print("Previous 2nd context vector:", context_vec_2)
