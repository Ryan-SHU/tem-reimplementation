"""
Family tree graph environment.

The tree has 3 generations with the structure from Whittington et al. (2020):

Generation 0 (grandparents):   nodes  0, 1
Generation 1 (parents):        nodes  2, 3, 4, 5
Generation 2 (children):       nodes  6, 7, 8, 9, 10, 11

12 nodes total.

Relations (actions):
    0 = parent      (child  -> parent)
    1 = child_left  (parent -> left child)
    2 = child_right (parent -> right child)
    3 = sibling     (node   -> its sibling)
    4 = partner     (parent -> partner = sibling's spouse)
    5 = self        (stay)

If a relation is undefined for a node, transition stays in place.

Paper reference: Figure 3 B, E  —  family tree hierarchy.
"""

from typing import Optional

import torch

from tem_data.base import GraphEnvironment


# Hardcoded tree topology
# Each entry: (left_child, right_child) or None if leaf
_CHILDREN = {
    0: (2, 3),
    1: (4, 5),
    2: (6, 7),
    3: (8, 9),
    4: (10, 11),
    5: None,
    6: None,
    7: None,
    8: None,
    9: None,
    10: None,
    11: None,
}

# Parent lookup
_PARENT = {}
for p, ch in _CHILDREN.items():
    if ch is not None:
        _PARENT[ch[0]] = p
        _PARENT[ch[1]] = p

# Sibling lookup
_SIBLING = {}
for p, ch in _CHILDREN.items():
    if ch is not None:
        _SIBLING[ch[0]] = ch[1]
        _SIBLING[ch[1]] = ch[0]

# Partner: defined for generation-1 parents
# Partners are the spouses from the other family branch
# Convention from the paper: nodes 2&4 are partners, 3&5 are partners
_PARTNER = {2: 4, 4: 2, 3: 5, 5: 3}


class FamilyTreeEnvironment(GraphEnvironment):

    name = "family_tree"

    NUM_NODES = 12
    NUM_ACTIONS = 6  # parent, child_left, child_right, sibling, partner, self
    ACTION_NAMES = [
        "parent", "child_left", "child_right", "sibling", "partner", "self",
    ]

    def __init__(
        self,
        num_observations: int,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.num_states = self.NUM_NODES
        self.num_actions = self.NUM_ACTIONS
        self.num_observations = num_observations

        self.transition_table = self._build_transitions()
        self.valid_action_mask = self._build_valid_mask()
        self.resample_observations(generator)

    def _build_transitions(self) -> torch.Tensor:
        S = self.num_states
        A = self.num_actions
        table = torch.zeros(S, A, dtype=torch.long)

        for s in range(S):
            # Default: self-loop
            table[s, :] = s

            # 0 = parent
            if s in _PARENT:
                table[s, 0] = _PARENT[s]

            # 1 = child_left
            if _CHILDREN.get(s) is not None:
                table[s, 1] = _CHILDREN[s][0]

            # 2 = child_right
            if _CHILDREN.get(s) is not None:
                table[s, 2] = _CHILDREN[s][1]

            # 3 = sibling
            if s in _SIBLING:
                table[s, 3] = _SIBLING[s]

            # 4 = partner
            if s in _PARTNER:
                table[s, 4] = _PARTNER[s]

            # 5 = self (already set)

        return table

    def _build_valid_mask(self) -> torch.Tensor:
        state_ids = torch.arange(self.num_states).unsqueeze(1)
        mask = self.transition_table != state_ids
        # Ensure at least one action is valid (self is always valid conceptually)
        for s in range(self.num_states):
            if not mask[s].any():
                mask[s, 5] = True  # self-loop is always "valid"
        return mask
