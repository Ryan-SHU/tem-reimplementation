# Data Generation Flow

This document explains how experimental data is generated before being passed into the TEM model.

The data pipeline has three layers:

```text
src/data/environments.py
    defines the environment structure

src/data/walks.py
    samples random walks in the environment

src/data/batches.py
    converts random walks into TEM-ready tensors
```

The TEM model does not directly know about grid coordinates, positions, or environment geometry.

The model only receives:

```text
x:       [B, T, N_x]
a:       [B, T, N_a]
visited: [B, T]
```

The extra fields such as `position`, `observation_id`, and `action_id` are saved for analysis.

---

# 1. Dimensions

```text
B   = batch size
T   = sequence length
H   = environment height
W   = environment width
S   = number of states = H * W
N_x = number of sensory observation categories
N_a = number of actions
```

For `RectangleEnvironment`:

```text
N_a = 4
```

The four actions are:

```text
0 = up
1 = down
2 = left
3 = right
```

---

# 2. Environment definition

The rectangular environment is a finite deterministic graph.

It contains:

```text
state space:
    S = {0, 1, ..., H * W - 1}

action space:
    A = {0, 1, 2, 3}

transition function:
    tau: S x A -> S

observation function:
    omega: S -> {0, 1, ..., N_x - 1}
```

In code, these correspond to:

```text
transition_table[s, a] = tau(s, a)

observation_ids[s] = omega(s)

valid_action_mask[s, a] = whether action a is valid at state s
```

---

# 3. State-coordinate mapping

The 2D grid cell `(row, col)` is mapped to a 1D state id by:

```text
state = row * width + col
```

For example, in a `3 x 4` grid:

```text
row 0: states  0,  1,  2,  3
row 1: states  4,  5,  6,  7
row 2: states  8,  9, 10, 11
```

So:

```text
(row=0, col=0) -> state 0
(row=0, col=1) -> state 1
(row=1, col=0) -> state 4
(row=2, col=3) -> state 11
```

The inverse mapping is:

```text
row = state // width
col = state % width
```

---

# 4. Transition function

The environment transition function is deterministic.

For each state `s` and action `a`:

```text
next_state = tau(s, a)
```

In code:

```python
next_state = environment.next_state(state, action)
```

For example, in a `3 x 4` grid, state `0` is the top-left corner.

```text
state 0 = row 0, col 0
```

Its transitions are:

```text
action 0 = up     -> wall, stay at 0
action 1 = down   -> move to state 4
action 2 = left   -> wall, stay at 0
action 3 = right  -> move to state 1
```

So:

```text
tau(0, 0) = 0
tau(0, 1) = 4
tau(0, 2) = 0
tau(0, 3) = 1
```

The transition table stores all such transitions:

```text
transition_table: [S, N_a]
```

---

# 5. Valid actions

The transition table includes wall collisions.

However, the random walk sampler usually avoids wall collisions.

So we define:

```text
valid_action_mask[s, a] = True
```

when action `a` actually moves the agent away from state `s`.

For the top-left corner:

```text
valid actions = [down, right] = [1, 3]
```

In code:

```python
valid_actions = environment.valid_actions_for_state(0)
```

returns:

```text
[1, 3]
```

Special case:

```text
If the environment is 1 x 1, no action can move the agent.
In that degenerate case, all actions are treated as valid.
```

---

# 6. Observation function

Each hidden environment state has a sensory observation id.

```text
omega(s) = observation_id
```

In code:

```text
observation_ids[s] = omega(s)
```

Shape:

```text
observation_ids: [S]
```

Example:

```text
state:          0  1  2  3  4  ...
observation:   7  2  9  1  5  ...
```

The model never receives the true state id directly.

Instead, the model receives the one-hot encoding of the observation id.

```text
x_t = one_hot(omega(s_t))
```

This is important:

```text
position/state = hidden environment variable
observation    = sensory input given to TEM
```

---

# 7. Random walk generation

The function:

```python
generate_random_walk(environment, batch_size, sequence_length, generator)
```

generates integer trajectories.

It returns:

```text
position: [B, T]
action:   [B, T]
visited:  [B, T]
```

No one-hot encoding happens here.

This function only works with discrete integer ids.

---

## 7.1 Start state

For each batch item `b`:

```text
s[b, 0] ~ Uniform({0, ..., S - 1})
```

In code:

```python
current_position = environment.sample_start_states(batch_size)
position[:, 0] = current_position
```

---

## 7.2 Dummy first action

At time step `t = 0`, there is no previous movement.

So the first action is set to a dummy value:

```text
action[b, 0] = 0
```

This means:

```text
a[:, 0, :] = one_hot(0)
```

This dummy action is mainly a placeholder so that `a` has shape `[B, T, N_a]`.

The model starts with zero recurrent state, so this first action has limited meaning.

---

## 7.3 Recursive walk rule

For each time step:

```text
t = 1, ..., T - 1
```

we sample an action from the valid actions at the previous state:

```text
a[b, t] ~ Uniform(valid_actions(s[b, t - 1]))
```

Then we apply the environment transition:

```text
s[b, t] = tau(s[b, t - 1], a[b, t])
```

In code:

```python
current_action = environment.sample_valid_actions(current_position)

next_position = environment.next_state(
    current_position,
    current_action,
)

action[:, t] = current_action
position[:, t] = next_position

current_position = next_position
```

---

# 8. Meaning of action timing

The convention is:

```text
position[:, t] = current state at time t
action[:, t]   = action used to arrive at position[:, t]
x[:, t]        = observation at position[:, t]
```

Therefore:

```text
action[:, 0] is dummy
action[:, 1] moves from position[:, 0] to position[:, 1]
action[:, 2] moves from position[:, 1] to position[:, 2]
...
```

This matches the TEM recurrent update:

```text
g_{t-1}, a_t -> g_t^{gen}
x_t          -> sensory evidence at current state
```

At step `t`, the model receives:

```text
a_t = action that brought the agent to the current state
x_t = sensory observation at the current state
```

---

# 9. Visited mask

The random walk currently sets:

```text
visited[b, t] = 1
```

for all batch items and all time steps.

Shape:

```text
visited: [B, T]
```

In TEM, `visited` controls memory update:

```text
visited[b, t] = 1
    update memory at time t

visited[b, t] = 0
    keep previous memory unchanged at time t
```

Currently, since every generated state is valid, every time step updates memory.

Later, for padded variable-length sequences, we may use:

```text
visited[b, t] = 0
```

for padded or invalid time steps.

---

# 10. Batch conversion

The class:

```python
RandomWalkBatcher
```

converts integer random walks into model-ready tensors.

It calls:

```python
walk = generate_random_walk(...)
```

The walk contains:

```text
position: [B, T]
action:   [B, T]
visited:  [B, T]
```

Then it computes:

```text
observation_id[b, t] = omega(position[b, t])
```

In code:

```python
observation_id = environment.get_observation_ids(position)
```

Then it converts integer ids to one-hot vectors:

```text
x[b, t] = one_hot(observation_id[b, t], N_x)
a[b, t] = one_hot(action[b, t], N_a)
```

So:

```text
x: [B, T, N_x]
a: [B, T, N_a]
```

---

# 11. Final batch dictionary

`RandomWalkBatcher.sample()` returns:

```python
batch = {
    "x": x,
    "a": a,
    "visited": visited,
    "position": position,
    "observation_id": observation_id,
    "action_id": action_id,
}
```

The first three fields are model inputs:

```text
x:       [B, T, N_x]
a:       [B, T, N_a]
visited: [B, T]
```

The other fields are for analysis:

```text
position:       [B, T]
observation_id: [B, T]
action_id:      [B, T]
```

TEM receives:

```python
output = model(
    x=batch["x"],
    a=batch["a"],
    visited=batch["visited"],
)
```

Loss receives:

```python
losses = compute_tem_loss(
    output=output,
    target_x=batch["x"],
    loss_config=config.loss,
)
```

---

# 12. Full data flow

The full data-generation pipeline is:

```text
RectangleEnvironment
    |
    | defines:
    |   transition_table
    |   valid_action_mask
    |   observation_ids
    |
    v
generate_random_walk
    |
    | produces integer trajectories:
    |   position: [B, T]
    |   action:   [B, T]
    |   visited:  [B, T]
    |
    v
RandomWalkBatcher
    |
    | maps:
    |   position -> observation_id
    |   observation_id -> one-hot x
    |   action -> one-hot a
    |
    v
batch
    |
    | model inputs:
    |   x
    |   a
    |   visited
    |
    v
TEM model
```

---

# 13. Mathematical summary

For each batch item `b`:

Initial state:

```text
s_{b,0} ~ Uniform(S)
```

Dummy action:

```text
a_{b,0} = 0
```

For time steps:

```text
t = 1, ..., T - 1
```

sample:

```text
a_{b,t} ~ Uniform(V(s_{b,t-1}))
```

transition:

```text
s_{b,t} = tau(s_{b,t-1}, a_{b,t})
```

observation:

```text
o_{b,t} = omega(s_{b,t})
```

one-hot sensory input:

```text
x_{b,t} = one_hot(o_{b,t}, N_x)
```

one-hot action input:

```text
A_{b,t} = one_hot(a_{b,t}, N_a)
```

visited mask:

```text
visited_{b,t} = 1
```

So the final model input is:

```text
x       = {x_{b,t}}       with shape [B, T, N_x]
a       = {A_{b,t}}       with shape [B, T, N_a]
visited = {visited_{b,t}} with shape [B, T]
```

---

# 14. What each file is responsible for

## environments.py

Responsible for environment structure.

It answers:

```text
How many states are there?
What are the possible actions?
Given state s and action a, what is the next state?
Given state s, what sensory observation is seen?
Which actions are valid at state s?
```

It should not know about:

```text
batch size
sequence length
TEM model
loss
training loop
```

---

## walks.py

Responsible for trajectory sampling.

It answers:

```text
Starting from random states, what sequence of states and actions does the agent experience?
```

It produces:

```text
position: [B, T]
action:   [B, T]
visited:  [B, T]
```

It should not perform one-hot encoding.

---

## batches.py

Responsible for converting trajectories into neural-network inputs.

It answers:

```text
How do we convert integer environment trajectories into tensors usable by TEM?
```

It produces:

```text
x:       [B, T, N_x]
a:       [B, T, N_a]
visited: [B, T]
```

It also keeps analysis metadata:

```text
position
observation_id
action_id
```

---

# 15. Why data is separate from tem

The `tem` package should define the model and loss.

The `data` package should define experimental worlds.

This separation is useful because the same TEM model can later be trained on different environments:

```text
RectangleEnvironment
HexagonalEnvironment
FamilyTreeEnvironment
LineTIEnvironment
```

without changing the model code.

The model should only care about tensor shapes:

```text
x:       [B, T, N_x]
a:       [B, T, N_a]
visited: [B, T]
```

The environment decides what those tensors mean.
