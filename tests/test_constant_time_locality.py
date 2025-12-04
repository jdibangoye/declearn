# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from declearn.core.tabular_tpi import TabularTPI
from declearn.core.types import make_sequential_dag
from declearn.core.sa_schedules import ThreeTimeScale
import numpy as np

def test_constant_time_update():
    dag = make_sequential_dag(2, 3)
    n_actions = {u.idx: 3 for u in dag.reverse_order()}
    tpi = TabularTPI(dag, n_actions, ThreeTimeScale(0.5,0.1,0.02), 1, np.random.default_rng(0))
    sample_dict = {}
    for u in dag.reverse_order():
        sample_dict[u.idx] = {
            "u": u.idx,
            "theta": ("s", u.time),
            "a": 0,
            "r": 0.0,
            "theta_next": ("s", u.time+1),
            "h": (("o",),()),
            "h_next": (("o2",),()),
            "a_next": 0,
        }
    tpi.step(sample_dict, 0)
    # if we reach here without error, it's ok