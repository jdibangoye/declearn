# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
# Placeholder adapter – actual rware may not be installed.
from typing import Any

class RWAREWrapper:
    def __init__(self, layout: str = "tiny-2ag", horizon: int = 50, seed: int = 0):
        self.layout = layout
        self.horizon = horizon
        self.seed = seed
        self.n_agents = 2  # usually
        self.n_actions = {0: 5, 1: 5}

    def reset(self):
        return {}

    def step(self, actions):
        return {}, {0: 0.0, 1: 0.0}, {0: False, 1: False}, {}