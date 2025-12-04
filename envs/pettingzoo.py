# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from typing import Any, Dict, Tuple, List, Optional
import numpy as np

try:
    from pettingzoo.mpe import simple_spread_v3, simple_tag_v3, simple_adversary_v3
    PZ_AVAILABLE = True
except ImportError:
    PZ_AVAILABLE = False

class PZWrapper:
    """
    Discrete wrapper over PettingZoo MPE tasks.
    We discretize observations if needed and expose per-agent obs.
    """
    def __init__(self, env_name: str, horizon: int = 25, seed: int = 0):
        assert PZ_AVAILABLE, "PettingZoo MPE not installed"
        if env_name == "simple_spread_v3":
            self.env = simple_spread_v3.parallel_env(max_cycles=horizon)
        elif env_name == "simple_tag_v3":
            self.env = simple_tag_v3.parallel_env(max_cycles=horizon)
        elif env_name == "simple_adversary_v3":
            self.env = simple_adversary_v3.parallel_env(max_cycles=horizon)
        else:
            raise ValueError(f"Unknown PZ env {env_name}")
        self.env.seed(seed)
        self.horizon = horizon
        self.agents = self.env.possible_agents
        self.n_agents = len(self.agents)
        # assume discrete action spaces
        self.n_actions = {i: self.env.action_space(self.agents[i]).n for i in range(self.n_agents)}

    def reset(self):
        obs = self.env.reset(seed=None)
        return obs

    def step(self, actions: Dict[str, int]):
        obs, rews, dones, infos = self.env.step(actions)
        return obs, rews, dones, infos