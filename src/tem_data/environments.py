"""
Backward-compatible environment exports.

New code should prefer importing from:

    tem_data.envs.rectangle
"""

from tem_data.envs.rectangle import RectangleEnvironment

__all__ = [
    "RectangleEnvironment",
]
