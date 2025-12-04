# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import argparse
import yaml
import numpy as np
from pathlib import Path

from declearn.envs.registry import build_env
from declearn.core.types import make_sequential_dag, priv_projection
from declearn.core.tabular_tpi import TabularTPI
from declearn.core.deep_tpi import DeepTPI
from declearn.core.sa_schedules import ThreeTimeScale
from declearn.evaluation.logging import ExperimentLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg.get("seed", 0))

    env = build_env(cfg["env"]["name"], **cfg["env"].get("kwargs", {}))

    # NEW: show summary if DecPOMDPSpec
    try:
        from declearn.envs.masplan import DecPOMDPSpec
        if isinstance(env, DecPOMDPSpec):
            print(env.summary())
    except Exception:
        pass

    n_agents = cfg["env"].get("n_agents", 2)
    horizon = cfg["env"].get("horizon", 10)
    dag = make_sequential_dag(n_agents, horizon)

    ts = ThreeTimeScale(
        alpha=cfg["algo"]["alpha"],
        beta=cfg["algo"]["beta"],
        gamma=cfg["algo"]["gamma"],
    )

    # n_actions per substage: for simplicity assume same across
    n_actions_per_u = {u.idx: cfg["env"]["n_actions"] for u in dag.reverse_order()}

    if cfg["algo"]["name"] == "tpi_tabular":
        agent = TabularTPI(
            dag=dag,
            n_actions_per_u=n_actions_per_u,
            timescale=ts,
            m_memory=cfg["algo"].get("m", 1),
            rng=rng,
        )
    elif cfg["algo"]["name"] == "tpi_deep":
        obs_dim_per_u = {u.idx: cfg["env"]["obs_dim"] for u in dag.reverse_order()}
        hist_dim_per_u = {u.idx: cfg["env"]["hist_dim"] for u in dag.reverse_order()}
        agent = DeepTPI(
            dag=dag,
            obs_dim_per_u=obs_dim_per_u,
            hist_dim_per_u=hist_dim_per_u,
            n_actions_per_u=n_actions_per_u,
            timescale=ts,
            m_memory=cfg["algo"].get("m", 1),
            device=cfg["algo"].get("device", "cpu"),
            seed=cfg.get("seed", 0),
        )
    else:
        raise ValueError("Unknown algo")

    out_dir = cfg.get("out_dir", "runs/default")
    logger = ExperimentLogger(out_dir, config=cfg)

    # training loop (simplified, on-policy single sample per substage)
    n_steps = cfg.get("train_steps", 1000)
    for k in range(n_steps):
        sample_dict = {}
        # we should get one sample per u from env/simulator; here we just mock
        for u in dag.reverse_order():
            uidx = u.idx
            theta = ("s0", u.time)  # placeholder
            h = (("o0",), ())
            theta_next = ("s1", u.time + 1)
            h_next = (("o1",), ())
            a = rng.integers(0, cfg["env"]["n_actions"])
            a_next = rng.integers(0, cfg["env"]["n_actions"])
            r = 0.0
            sample_dict[uidx] = {
                "u": uidx,
                "theta": theta,
                "a": a,
                "r": r,
                "theta_next": theta_next,
                "h": h,
                "h_next": h_next,
                "a_next": a_next,
            }
        agent.step(sample_dict, k)
        if (k + 1) % cfg.get("eval_every", 100) == 0:
            logger.log_eval(mean=0.0, ci=0.0)

if __name__ == "__main__":
    main()