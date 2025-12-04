# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from typing import Dict, List, Any
import numpy as np

class OnPolicyBuffer:
    """
    Stores samples tagged by sub-stage u.
    Each entry: (theta, a, r, theta_next).
    """
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.data: Dict[int, List[Any]] = {}

    def add(self, uidx: int, sample: Any):
        if uidx not in self.data:
            self.data[uidx] = []
        if len(self.data[uidx]) >= self.capacity:
            self.data[uidx].pop(0)
        self.data[uidx].append(sample)

    def sample_recent(self, uidx: int):
        if uidx not in self.data or len(self.data[uidx]) == 0:
            return None
        return self.data[uidx][-1]