# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from declearn.evaluation.logging import ExperimentLogger
from declearn.evaluation.metrics import mean_ci
import os
import numpy as np

def test_eval_log():
    logger = ExperimentLogger("runs/tests", {"dummy": 1})
    logger.log_eval(mean=1.0, ci=0.1)
    assert os.path.exists("runs/tests/results.csv")

def test_ci():
    m, c = mean_ci(np.array([1.0, 1.0, 1.0]))
    assert m == 1.0
    assert c >= 0.0