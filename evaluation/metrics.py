# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import numpy as np
from typing import Tuple

def mean_ci(x: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    mean = float(np.mean(x))
    # simple normal approx
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    # 1.96 ok
    ci = 1.96 * se
    return mean, ci

def auc_curve(y: np.ndarray) -> float:
    # trapezoidal
    return float(np.trapz(y))

def time_to_threshold(y: np.ndarray, tau: float) -> int:
    for i, v in enumerate(y):
        if v >= tau:
            return i
    return len(y)