# TEM Computation Flow

This document freezes the tensor shapes and forward computation order for the first implementation.

All tensors use PyTorch convention.

```text
batch dimension first
time dimension second
module-specific tensors stored as list[Tensor]
```

---

## 1. Global dimensions

```text
B   = batch size
T   = sequence length
N_x = number of sensory observation categories
N_a = number of action / relation categories
F   = number of modules
C   = compressed sensory dimension
K   = number of attractor iterations
```

For module `f`:

```text
G_f   = grid / abstract-location dimension
Phi_f = phase dimension
P_f   = Phi_f * C
```

---

## 2. Inputs

```text
x:       [B, T, N_x]   float32
a:       [B, T, N_a]   float32
visited: [B, T]        float32 or bool
```

`x` is one-hot or probability-like sensory observation.

`a` is one-hot action / relation input.

`visited` controls memory update:

```text
visited[b, t] = 1  update memory at step t
visited[b, t] = 0  keep previous memory unchanged
```

---

## 3. Recurrent state

At the beginning of step `t`, for each module `f`:

```text
g_prev[f]:          [B, G_f]
x_s_prev[f]:        [B, C]
memory_gen_prev[f]: [B, P_f, P_f]
memory_inf_prev[f]: [B, P_f, P_f]
```

If no initial state is provided:

```text
g_prev[f]          = zeros([B, G_f])
x_s_prev[f]        = zeros([B, C])
memory_gen_prev[f] = zeros([B, P_f, P_f])
memory_inf_prev[f] = zeros([B, P_f, P_f])
```

---

## 4. Parameters

### 4.1 Sensory encoder

```text
W_xc: [N_x, C]
b_xc: [C]
```

### 4.2 Transition model

For each module `f`:

```text
W_D[f]:          [N_a, G_f * G_f]
b_D[f]:          [G_f * G_f]

W_sigma_path[f]: [N_a, G_f]
b_sigma_path[f]: [G_f]
```

### 4.3 Memory-to-grid inference

For each module `f`:

```text
W_mu_mem[f]:     [Phi_f, G_f]
b_mu_mem[f]:     [G_f]

W_sigma_mem[f]:  [Phi_f, G_f]
b_sigma_mem[f]:  [G_f]
```

### 4.4 Grid-to-phase projection

For each module `f`:

```text
W_g2phi[f]: [G_f, Phi_f]
b_g2phi[f]: [Phi_f]
```

### 4.5 Decoder

```text
W_dec: [F * C, N_x]
b_dec: [N_x]
```

### 4.6 Scalars

For each module `f`:

```text
alpha[f]: [scalar]
gamma[f]: [scalar]
kappa[f]: [scalar]
```

Global memory scalars:

```text
memory_eta_gen:    [scalar]
memory_eta_inf:    [scalar]
memory_lambda_gen: [scalar]
memory_lambda_inf: [scalar]
memory_clip:       [scalar]
```

---

## 5. Basic functions

### 5.1 Grid activation

Input:

```text
z: any shape
```

Output:

```text
phi_g(z): same shape
```

Definition:

```text
phi_g(z) = clamp(z, min=-1, max=1)
```

Matches original paper: `tf.minimum(tf.maximum(g, -1), 1)` (no tanh).

---

### 5.2 Place activation

Input:

```text
z: any shape
```

Output:

```text
phi_p(z): same shape
```

Definition:

```text
phi_p(z) = leaky_relu(clamp(z, min=-1, max=1))
```

Matches original paper: `tf.nn.leaky_relu(tf.minimum(tf.maximum(p, -1), 1))`.
Note: clamp is applied FIRST, then leaky_relu.

---

### 5.3 Normalize

Input:

```text
u: [B, C]
```

Output:

```text
normalize(u): [B, C]
```

Definition:

```text
normalize(u) = u / (||u||_2 + eps)
```

---

### 5.4 Tile sensory vector across phases

Input:

```text
u: [B, C]
```

Output:

```text
tile_phase(u, Phi_f): [B, Phi_f * C]
```

Definition:

```text
tile_phase(u, Phi_f)
= repeat u for Phi_f phases
```

Shape:

```text
[B, C] -> [B, Phi_f, C] -> [B, P_f]
```

---

### 5.5 Repeat phase vector across sensory dimension

Input:

```text
r: [B, Phi_f]
```

Output:

```text
repeat_sensory(r, C): [B, Phi_f * C]
```

Definition:

```text
repeat_sensory(r, C)
= repeat each phase value over C sensory channels
```

Shape:

```text
[B, Phi_f] -> [B, Phi_f, C] -> [B, P_f]
```

---

### 5.6 Sum over phase dimension

Input:

```text
p_f: [B, P_f]
```

Reshape:

```text
p_f_view: [B, Phi_f, C]
```

Output:

```text
phase_sum(p_f): [B, C]
```

Definition:

```text
phase_sum(p_f) = sum over Phi_f
```

---

### 5.7 Sum over sensory dimension

Input:

```text
p_f: [B, P_f]
```

Reshape:

```text
p_f_view: [B, Phi_f, C]
```

Output:

```text
sense_sum(p_f): [B, Phi_f]
```

Definition:

```text
sense_sum(p_f) = sum over C
```

---

## 6. Attractor retrieval

For module `f`.

Input:

```text
query:  [B, P_f]
memory: [B, P_f, P_f]
```

Initialize:

```text
h_0 = phi_p(query)
h_0: [B, P_f]
```

For `k = 0, ..., K - 1`:

```text
memory_drive_k = bmm(h_k[:, None, :], memory)[:, 0, :]
memory_drive_k: [B, P_f]
```

```text
h_{k+1} = phi_p(kappa[f] * h_k + memory_drive_k)
h_{k+1}: [B, P_f]
```

Output:

```text
attractor(query, memory, f) = h_K
h_K: [B, P_f]
```

---

## 7. One-step forward computation

For each time step:

```text
t = 0, ..., T - 1
```

---

### 7.1 Read current inputs

```text
x_t = x[:, t, :]
x_t: [B, N_x]
```

```text
a_t = a[:, t, :]
a_t: [B, N_a]
```

```text
visited_t = visited[:, t]
visited_t: [B]
```

```text
visited_memory_mask = visited_t.view(B, 1, 1)
visited_memory_mask: [B, 1, 1]
```

---

### 7.2 Sensory encoding

```text
x_c_t = phi_p(x_t @ W_xc + b_xc)
x_c_t: [B, C]
```

For each module `f`:

```text
x_s_t[f] =
    (1 - alpha[f]) * x_s_prev[f]
    + alpha[f] * x_c_t

x_s_t[f]: [B, C]
```

```text
x_s_mean_t[f] = mean(x_s_t[f], dim=-1, keepdim=True)
x_s_mean_t[f]: [B, 1]
```

```text
x_s_centered_t[f] = relu(x_s_t[f] - x_s_mean_t[f])
x_s_centered_t[f]: [B, C]
```

```text
x_s_norm_t[f] = normalize(x_s_centered_t[f])
x_s_norm_t[f]: [B, C]
```

```text
x_2p_t[f] = gamma[f] * tile_phase(x_s_norm_t[f], Phi_f)
x_2p_t[f]: [B, P_f]
```

---

### 7.3 Structural transition

For each module `f`:

```text
d_t[f] = a_t @ W_D[f] + b_D[f]
d_t[f]: [B, G_f * G_f]
```

```text
D_t[f] = d_t[f].view(B, G_f, G_f)
D_t[f]: [B, G_f, G_f]
```

```text
delta_g_t[f] = bmm(g_prev[f][:, None, :], D_t[f])[:, 0, :]
delta_g_t[f]: [B, G_f]
```

```text
mu_path_t[f] = phi_g(g_prev[f] + delta_g_t[f])
mu_path_t[f]: [B, G_f]
```

```text
sigma_path_t[f] = softplus(a_t @ W_sigma_path[f] + b_sigma_path[f]) + eps
sigma_path_t[f]: [B, G_f]
```

```text
g_gen_t[f] = mu_path_t[f]
g_gen_t[f]: [B, G_f]
```

---

### 7.4 Inference memory retrieval from sensory cue

For each module `f`:

```text
p_x_t[f] = attractor(
    query=x_2p_t[f],
    memory=memory_inf_prev[f],
    module=f,
)

p_x_t[f]: [B, P_f]
```

```text
u_x_t[f] = sense_sum(p_x_t[f])
u_x_t[f]: [B, Phi_f]
```

```text
mu_mem_t[f] = phi_g(u_x_t[f] @ W_mu_mem[f] + b_mu_mem[f])
mu_mem_t[f]: [B, G_f]
```

```text
sigma_mem_t[f] = softplus(u_x_t[f] @ W_sigma_mem[f] + b_sigma_mem[f]) + eps
sigma_mem_t[f]: [B, G_f]
```

---

### 7.5 Combine path estimate and memory estimate

For each module `f`:

```text
var_path_t[f] = sigma_path_t[f] ** 2
var_path_t[f]: [B, G_f]
```

```text
var_mem_t[f] = sigma_mem_t[f] ** 2
var_mem_t[f]: [B, G_f]
```

```text
precision_path_t[f] = 1 / (var_path_t[f] + eps)
precision_path_t[f]: [B, G_f]
```

```text
precision_mem_t[f] = 1 / (var_mem_t[f] + eps)
precision_mem_t[f]: [B, G_f]
```

```text
precision_total_t[f] =
    precision_path_t[f]
    + precision_mem_t[f]

precision_total_t[f]: [B, G_f]
```

```text
mu_g_t[f] =
    (
        mu_path_t[f] * precision_path_t[f]
        + mu_mem_t[f] * precision_mem_t[f]
    )
    / (precision_total_t[f] + eps)

mu_g_t[f]: [B, G_f]
```

```text
g_t[f] = phi_g(mu_g_t[f])
g_t[f]: [B, G_f]
```

---

### 7.6 Project grid state to p-space

For each module `f`:

```text
r_t[f] = phi_g(g_t[f] @ W_g2phi[f] + b_g2phi[f])
r_t[f]: [B, Phi_f]
```

```text
g_2p_t[f] = repeat_sensory(r_t[f], C)
g_2p_t[f]: [B, P_f]
```

Also for transition-only generated grid state:

```text
r_gen_t[f] = phi_g(g_gen_t[f] @ W_g2phi[f] + b_g2phi[f])
r_gen_t[f]: [B, Phi_f]
```

```text
g_gen_2p_t[f] = repeat_sensory(r_gen_t[f], C)
g_gen_2p_t[f]: [B, P_f]
```

---

### 7.7 Infer bound place representation

For each module `f`:

```text
p_t[f] = phi_p(g_2p_t[f] * x_2p_t[f])
p_t[f]: [B, P_f]
```

---

### 7.8 Generative memory retrieval from inferred grid state

For each module `f`:

```text
p_g_t[f] = attractor(
    query=g_2p_t[f],
    memory=memory_gen_prev[f],
    module=f,
)

p_g_t[f]: [B, P_f]
```

---

### 7.9 Generative memory retrieval from transition-only grid state

For each module `f`:

```text
p_gt_t[f] = attractor(
    query=g_gen_2p_t[f],
    memory=memory_gen_prev[f],
    module=f,
)

p_gt_t[f]: [B, P_f]
```

---

### 7.10 Decode predictions

For each module `f`:

```text
z_p_t[f] = phase_sum(p_t[f])
z_p_t[f]: [B, C]
```

```text
z_g_t[f] = phase_sum(p_g_t[f])
z_g_t[f]: [B, C]
```

```text
z_gt_t[f] = phase_sum(p_gt_t[f])
z_gt_t[f]: [B, C]
```

Concatenate modules:

```text
z_p_t = concat(z_p_t[0], ..., z_p_t[F - 1], dim=-1)
z_p_t: [B, F * C]
```

```text
z_g_t = concat(z_g_t[0], ..., z_g_t[F - 1], dim=-1)
z_g_t: [B, F * C]
```

```text
z_gt_t = concat(z_gt_t[0], ..., z_gt_t[F - 1], dim=-1)
z_gt_t: [B, F * C]
```

Logits:

```text
x_p_logits_t = z_p_t @ W_dec + b_dec
x_p_logits_t: [B, N_x]
```

```text
x_g_logits_t = z_g_t @ W_dec + b_dec
x_g_logits_t: [B, N_x]
```

```text
x_gt_logits_t = z_gt_t @ W_dec + b_dec
x_gt_logits_t: [B, N_x]
```

Probabilities:

```text
x_p_t = softmax(x_p_logits_t, dim=-1)
x_p_t: [B, N_x]
```

```text
x_g_t = softmax(x_g_logits_t, dim=-1)
x_g_t: [B, N_x]
```

```text
x_gt_t = softmax(x_gt_logits_t, dim=-1)
x_gt_t: [B, N_x]
```

---

### 7.11 Hebbian memory update

For each module `f`.

Generative memory update:

```text
a_gen_t[f] = p_t[f] - p_g_t[f]
a_gen_t[f]: [B, P_f]
```

```text
b_gen_t[f] = p_t[f] + p_g_t[f]
b_gen_t[f]: [B, P_f]
```

```text
delta_memory_gen_t[f] =
    b_gen_t[f][:, :, None]
    * a_gen_t[f][:, None, :]

delta_memory_gen_t[f]: [B, P_f, P_f]
```

```text
memory_gen_candidate_t[f] =
    clamp(
        memory_lambda_gen * memory_gen_prev[f]
        + memory_eta_gen * delta_memory_gen_t[f],
        min=-memory_clip,
        max=memory_clip,
    )

memory_gen_candidate_t[f]: [B, P_f, P_f]
```

Apply visited mask:

```text
memory_gen_t[f] =
    visited_memory_mask * memory_gen_candidate_t[f]
    + (1 - visited_memory_mask) * memory_gen_prev[f]

memory_gen_t[f]: [B, P_f, P_f]
```

Inference memory update:

```text
a_inf_t[f] = p_t[f] - p_x_t[f]
a_inf_t[f]: [B, P_f]
```

```text
b_inf_t[f] = p_t[f] + p_x_t[f]
b_inf_t[f]: [B, P_f]
```

```text
delta_memory_inf_t[f] =
    b_inf_t[f][:, :, None]
    * a_inf_t[f][:, None, :]

delta_memory_inf_t[f]: [B, P_f, P_f]
```

```text
memory_inf_candidate_t[f] =
    clamp(
        memory_lambda_inf * memory_inf_prev[f]
        + memory_eta_inf * delta_memory_inf_t[f],
        min=-memory_clip,
        max=memory_clip,
    )

memory_inf_candidate_t[f]: [B, P_f, P_f]
```

Apply visited mask:

```text
memory_inf_t[f] =
    visited_memory_mask * memory_inf_candidate_t[f]
    + (1 - visited_memory_mask) * memory_inf_prev[f]

memory_inf_t[f]: [B, P_f, P_f]
```

---

### 7.12 Carry recurrent state to next time step

For each module `f`:

```text
g_prev[f] = g_t[f]
g_prev[f]: [B, G_f]
```

```text
x_s_prev[f] = x_s_t[f]
x_s_prev[f]: [B, C]
```

```text
memory_gen_prev[f] = memory_gen_t[f]
memory_gen_prev[f]: [B, P_f, P_f]
```

```text
memory_inf_prev[f] = memory_inf_t[f]
memory_inf_prev[f]: [B, P_f, P_f]
```

---

## 8. Sequence outputs

After looping over all time steps, stack per-step outputs along time dimension.

### 8.1 Observation predictions

```text
x_p_logits:  [B, T, N_x]
x_g_logits:  [B, T, N_x]
x_gt_logits: [B, T, N_x]
```

```text
x_p:  [B, T, N_x]
x_g:  [B, T, N_x]
x_gt: [B, T, N_x]
```

---

### 8.2 Module-wise latent outputs

For each module `f`:

```text
g[f]:     [B, T, G_f]
g_gen[f]: [B, T, G_f]
```

```text
p[f]:    [B, T, P_f]
p_x[f]:  [B, T, P_f]
p_g[f]:  [B, T, P_f]
p_gt[f]: [B, T, P_f]
```

```text
x_s[f]: [B, T, C]
```

---

### 8.3 Final recurrent state

For each module `f`:

```text
final_state.g[f]:          [B, G_f]
final_state.x_s[f]:        [B, C]
final_state.memory_gen[f]: [B, P_f, P_f]
final_state.memory_inf[f]: [B, P_f, P_f]
```

---

## 9. Full one-step dependency summary

```text
x_t
    -> x_c_t
    -> x_s_t[f]
    -> x_2p_t[f]

g_prev[f], a_t
    -> g_gen_t[f]

x_2p_t[f], memory_inf_prev[f]
    -> p_x_t[f]
    -> u_x_t[f]
    -> mu_mem_t[f], sigma_mem_t[f]

g_gen_t[f], mu_mem_t[f], sigma_path_t[f], sigma_mem_t[f]
    -> g_t[f]

g_t[f]
    -> r_t[f]
    -> g_2p_t[f]

g_gen_t[f]
    -> r_gen_t[f]
    -> g_gen_2p_t[f]

g_2p_t[f], x_2p_t[f]
    -> p_t[f]

g_2p_t[f], memory_gen_prev[f]
    -> p_g_t[f]

g_gen_2p_t[f], memory_gen_prev[f]
    -> p_gt_t[f]

p_t[f]
    -> z_p_t[f]
    -> x_p_logits_t
    -> x_p_t

p_g_t[f]
    -> z_g_t[f]
    -> x_g_logits_t
    -> x_g_t

p_gt_t[f]
    -> z_gt_t[f]
    -> x_gt_logits_t
    -> x_gt_t

p_t[f], p_g_t[f], memory_gen_prev[f], visited_t
    -> memory_gen_t[f]

p_t[f], p_x_t[f], memory_inf_prev[f], visited_t
    -> memory_inf_t[f]
```

---

## 10. Implementation order

The code should be implemented in this order:

```text
1. activations.py
2. tensor_ops.py
3. attractor.py
4. state.py
5. sensory.py
6. transition.py
7. inference.py
8. decoder.py
9. memory.py
10. tem.py
11. losses.py
12. train.py
```
