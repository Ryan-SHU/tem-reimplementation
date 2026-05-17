"""
Experiment 1B: Structural generalization on a family tree.

Paper reference: Figure 3 B, E

Graph: 12 nodes, 6 relations (parent, child_left, child_right, sibling, partner, self).
"""

NUM_NODES = 12
NUM_ACTIONS = 6
NUM_OBSERVATIONS = 45
TRAIN_SEQ_LEN = 50
EVAL_EXPLORE_LEN = 20
TRAIN_BATCH_SIZE = 16
TRAIN_STEPS = 10000
EVAL_EPISODES = 200
