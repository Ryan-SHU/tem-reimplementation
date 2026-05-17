"""
Experiment 1A: Transitive Inference on a line graph.

Paper reference
---------------
Main text:  "TEM generalises structural knowledge …", Figure 3 A, D
STAR Methods: "Transitive inference" paragraph

Graph
-----
    0 -- 1 -- 2 -- ... -- (N-1)

    num_nodes  = 7   (paper default)
    num_actions = 2   (left=0, right=1)

Dimensions
----------
    B       batch size
    T       sequence length (training walk)
    T_exp   exploration length (evaluation walk in new world)
    N_x     number of distinct sensory observations
    N_a     2
    S       7

Training protocol
-----------------
    For each training step:
        1. env.resample_observations()   # new world
        2. walk = random_walk(env, B, T)
        3. batch = one_hot(walk)
        4. output = model(batch)
        5. loss   = compute_tem_loss(output, batch["x"])
        6. optimizer.step()

Evaluation protocol
-------------------
    In a new world:
        1. env.resample_observations()
        2. exploration walk of length T_exp
        3. Feed walk to model (build memory)
        4. For each *un-traversed* valid edge (s, a, s'):
             - teleport model to state s (by feeding x_s with dummy action)
             - apply action a
             - check if argmax of predicted x equals observation at s'
        5. Report zero-shot accuracy

Key metric
----------
    zero_shot_accuracy =
        (# correct predictions on un-traversed edges) /
        (# un-traversed edges queried)
"""

# Default hyper-parameters following the paper
NUM_NODES = 7
NUM_ACTIONS = 2
NUM_OBSERVATIONS = 45      # paper uses N_x = 45 for TI
TRAIN_SEQ_LEN = 30         # T during training
EVAL_EXPLORE_LEN = 12      # T_exp: partial exploration
TRAIN_BATCH_SIZE = 16
TRAIN_STEPS = 10000
EVAL_EPISODES = 200        # number of new-world episodes for evaluation
