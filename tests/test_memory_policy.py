# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import numpy as np
from declearn.core.memory_policy import MemoryPolicy

def test_mirror_step_keeps_simplex():
    pol = MemoryPolicy(n_actions=3)
    h = ("o",)
    pol.mirror_update(0, h, np.array([1.0, 0.0, 0.0]))
    p = pol.probs(0, h)
    assert np.isclose(np.sum(p), 1.0)
    assert np.all(p > 0)