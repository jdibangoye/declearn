# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from typing import Callable, Dict, Any, List
import numpy as np

from .metrics import mean_ci, auc_curve, time_to_threshold
from .logging import ExperimentLogger

def run_eval(
    agent,
    env_builder: Callable[[], Any],
    episodes: int,
    greedy: bool,
    logger: ExperimentLogger,
    threshold: float = None,
):
    returns = []
    for _ in range(episodes):
        env = env_builder()
        obs = env.reset()
        done = False
        ep_ret = 0.0
        t = 0
        # this part is env-specific; for decpomdp we would need a simulator
        while not done and t < 100:
            # placeholder
            ep_ret += 0.0
            t += 1
            done = True
        returns.append(ep_ret)
    mean, ci = mean_ci(np.array(returns))
    logger.log_eval(mean=mean, ci=ci)
    if threshold is not None:
        ttt = time_to_threshold(np.array(returns), threshold)
        logger.log_eval(time_to_threshold=ttt)
    return mean, ci