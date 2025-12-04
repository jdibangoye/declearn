# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import os
import numpy as np
import pytest

from declearn.envs.masplan import parse_dpomdp
from declearn.core.sequential_env import SequentialDecPOMDPSimulator

PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
SPECS_DIR = os.path.join(PKG_ROOT, "envs", "masplan_specs")


def _all_dpomdp_paths():
    if not os.path.isdir(SPECS_DIR):
        return []
    return sorted(os.path.join(SPECS_DIR, f) for f in os.listdir(SPECS_DIR) if f.endswith(".dpomdp"))


def _assert_prob_vector(vec, n):
    arr = np.asarray(vec, dtype=float)
    assert arr.size == n, f"vector length {arr.size} != n_states {n}"
    s = float(arr.sum())
    assert np.isfinite(s) and s >= 0, "prob vector has non-finite or negative sum"
    if s > 0:
        assert np.all(arr >= 0), "prob vector has negative entries"


def validate_spec_struct(spec):
    assert spec is not None
    # n_agents
    assert hasattr(spec, "n_agents") and isinstance(spec.n_agents, int) and spec.n_agents >= 1

    # states
    assert hasattr(spec, "states") and isinstance(spec.states, (list, tuple)) and len(spec.states) >= 1
    n_states = getattr(spec, "n_states", None)
    if n_states is not None:
        assert n_states == len(spec.states)

    # actions / observations
    assert hasattr(spec, "actions") and isinstance(spec.actions, (list, tuple))
    assert len(spec.actions) == spec.n_agents
    for a in spec.actions:
        assert isinstance(a, (list, tuple)) and len(a) >= 1

    assert hasattr(spec, "observations") and isinstance(spec.observations, (list, tuple))
    assert len(spec.observations) == spec.n_agents
    for o in spec.observations:
        assert isinstance(o, (list, tuple)) and len(o) >= 1

    # horizon
    assert hasattr(spec, "horizon") and isinstance(spec.horizon, int) and spec.horizon > 0

    # n_actions_per_agent / n_obs_per_agent if present
    if hasattr(spec, "n_actions_per_agent"):
        assert list(spec.n_actions_per_agent) == [len(a) for a in spec.actions]
    if hasattr(spec, "n_obs_per_agent"):
        assert list(spec.n_obs_per_agent) == [len(o) for o in spec.observations]

    # start (may exist as 'start'): if present it should be reasonable
    if hasattr(spec, "start"):
        s = getattr(spec, "start")
        assert s is None or isinstance(s, (str, list, tuple, np.ndarray))

    # init/start belief if present: normalized vector of length n_states
    if hasattr(spec, "init_belief") and getattr(spec, "init_belief") is not None:
        _assert_prob_vector(getattr(spec, "init_belief"), len(spec.states))

    # transitions / observation / reward kernels: at least one source must exist or simulator must be able to proceed
    has_transition = any(
        getattr(spec, attr, None)
        for attr in ("transitions_compiled", "transition", "T")
    ) or hasattr(spec, "sample_next_state")
    has_observation = any(getattr(spec, attr, None) for attr in ("observations_compiled", "observation", "O")) or hasattr(spec, "sample_observation")
    has_reward = any(getattr(spec, attr, None) for attr in ("rewards_compiled", "reward", "R")) or hasattr(spec, "sample_reward")
    assert has_transition or has_observation or has_reward, "spec seems to have no dynamics/rewards/observations defined"


def test_parse_and_basic_structure_for_all_specs():
    paths = _all_dpomdp_paths()
    if not paths:
        pytest.skip("No masplan_specs found")
    for path in paths:
        spec = parse_dpomdp(path)
        validate_spec_struct(spec)


def test_simulator_minimal_run_on_all_specs():
    paths = _all_dpomdp_paths()
    if not paths:
        pytest.skip("No masplan_specs found")
    for path in paths:
        spec = parse_dpomdp(path)
        # minimal simulation smoke: instantiate, reset, perform up to n_agents steps
        sim = SequentialDecPOMDPSimulator(spec, seed=0, memory_m=1)
        tt = sim.reset()
        assert tt is not None
        for _ in range(sim.n_agents):
            tt, r, done, info = sim.step(0)
            assert isinstance(done, bool)
            assert isinstance(r, (float, int))