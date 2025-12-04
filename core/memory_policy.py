# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from typing import Dict, Tuple, List
import math
import numpy as np


class MemoryPolicy:
    """
    σ_u(h) with bounded memory m.
    We store logits per (u, h) and do mirror-ascent updates.
    """

    def __init__(self, n_actions: int, gamma_init: float = 0.1):
        self.n_actions = n_actions
        self.logits: Dict[Tuple[int, Tuple], np.ndarray] = {}
        self.gamma = gamma_init

    def _ensure(self, uidx: int, h: Tuple):
        key = (uidx, h)
        if key not in self.logits:
            self.logits[key] = np.zeros(self.n_actions, dtype=np.float32)

    def probs(self, uidx: int, h: Tuple) -> np.ndarray:
        self._ensure(uidx, h)
        z = self.logits[(uidx, h)]
        z = z - np.max(z)
        p = np.exp(z)
        return p / np.sum(p)

    def sample(self, uidx: int, h: Tuple, rng: np.random.Generator) -> int:
        p = self.probs(uidx, h)
        return int(rng.choice(self.n_actions, p=p))

    def greedy(self, uidx: int, h: Tuple) -> int:
        p = self.probs(uidx, h)
        return int(np.argmax(p))

    def mirror_update(self, uidx: int, h: Tuple, g_vals: np.ndarray, gamma: float = None):
        """
        Mirror ascent in KL geometry:
            w <- w + gamma * (g - <σ,g>)
            σ = softmax(w)
        g_vals: shape (n_actions,)
        """
        self._ensure(uidx, h)
        w = self.logits[(uidx, h)]
        sigma = self.probs(uidx, h)
        exp_val = float(np.sum(sigma * g_vals))
        step = gamma if gamma is not None else self.gamma
        w_new = w + step * (g_vals - exp_val)
        self.logits[(uidx, h)] = w_new