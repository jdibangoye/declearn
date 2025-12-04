# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
"""
Lightweight CTDE/value-decomposition wrappers.

We do NOT implement the full algorithms here, but provide the interfaces
expected by the evaluation protocol: fit(), evaluate().
Actual models can be plugged in by the user.
"""
from typing import Any, Dict, List
import numpy as np

class DummyVDN:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def fit(self, batch: Any):
        pass

    def evaluate(self, env: Any, episodes: int = 10, greedy: bool = True) -> float:
        return 0.0

def build_ctde(name: str, n_agents: int):
    if name.lower() == "vdn":
        return DummyVDN(n_agents)
    # extend with qmix, qtran, qplex stubs
    return DummyVDN(n_agents)