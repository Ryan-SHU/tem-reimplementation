"""
Experiment 1A: Transitive Inference on a line graph.

Paper reference
---------------
Main text: Figure 3 A, D
Task family: structural generalization in a new sensory environment

Graph
-----
    0 -- 1 -- 2 -- ... -- (N-1)

    num_nodes   = 7   (paper default)
    num_actions = 2   (left=0, right=1)

Key idea
--------
Training:
    The model is trained across many worlds that share the same graph
    structure but differ in their sensory assignment.

Evaluation:
    Evaluation is done offline, from saved checkpoints.
    For each checkpoint:
        1. create many fresh worlds
        2. run a continuous walk in each world
        3. at every step, record whether the current edge is:
             - seen before
             - unseen but inferable
        4. save raw event-level evaluation rows
        5. aggregate later for plotting

Paper-style plotting:
    x-axis:
        - number of linked observations
        - number of nodes visited
        - proportion of nodes visited

    y-axis:
        - correct inference of link

    different curves:
        - different training quantiles
"""

NUM_NODES = 7
NUM_ACTIONS = 2
NUM_OBSERVATIONS = 45

TRAIN_SEQ_LEN = 30
TRAIN_BATCH_SIZE = 16
TRAIN_STEPS = 10000

# evaluation uses a longer continuous walk
EVAL_EXPLORE_LEN = 12
EVAL_WALK_LEN = 36
EVAL_EPISODES = 200

NUM_TRAINING_QUANTILES = 5
DEFAULT_CHECKPOINT_EVERY = 500
