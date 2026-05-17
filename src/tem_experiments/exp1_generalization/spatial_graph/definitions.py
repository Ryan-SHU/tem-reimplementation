"""
Experiment 1C: Structural generalization on a 2D spatial graph.

Paper reference: Figure 3 C, F

Graph: height x width rectangle grid, 4 actions (up/down/left/right).
"""

HEIGHT = 6
WIDTH = 6
NUM_ACTIONS = 4
NUM_OBSERVATIONS = 45
TRAIN_SEQ_LEN = 50
EVAL_EXPLORE_LEN = 25
TRAIN_BATCH_SIZE = 16
TRAIN_STEPS = 10000
EVAL_EPISODES = 200
