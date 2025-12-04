# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import os
import numpy as np
from typing import Tuple
import pytest
from declearn.envs.masplan import parse_dpomdp
from declearn.core.sequential_env import SequentialDecPOMDPSimulator

SPEC_NAMES = [
    "dectiger.dpomdp",
    "broadcastChannel.dpomdp",
    "recycling.dpomdp",
    "boxPushingUAI07.dpomdp",
    "GridSmall.dpomdp",
]

class WrappedSpec:
    """Minimal wrapper exposing the fields expected by SequentialDecPOMDPSimulator."""
    def __init__(self, parsed):
        # basic attrs
        self.n_agents = getattr(parsed, "n_agents")
        # ensure horizon is an int fallbacking to parsed.H or 50 when None/missing
        raw_h = getattr(parsed, "horizon", None)
        if raw_h is None:
            raw_h = getattr(parsed, "H", None)
        if raw_h is None:
            raw_h = 10
        self.horizon = int(raw_h)
        self.n_states = getattr(parsed, "n_states")
        # actions / observations left as parsed (lists of names or counts)
        self.actions = getattr(parsed, "actions")
        self.observations = getattr(parsed, "observations")
        # init belief compatibility
        self.init_belief = (
            list(getattr(parsed, "start_belief"))
            if getattr(parsed, "start_belief", None) is not None
            else list(getattr(parsed, "init_belief", None) or [1.0 / self.n_states] * self.n_states)
        )
        # compile transitions: expect parsed.T keys like (s, a_tuple) -> probs
        self.transitions_compiled = {}
        T = getattr(parsed, "T", None)
        if T:
            for (s, a_tuple), probs in T.items():
                probs = np.asarray(probs, dtype=float)
                for sp, p in enumerate(probs):
                    if p:
                        self.transitions_compiled[(int(s), tuple(int(x) for x in a_tuple), int(sp))] = float(p)
        # compile observations: expect parsed.O keys like (s, a_tuple) -> arr (joint obs array)
        self.observations_compiled = {}
        O = getattr(parsed, "O", None)
        if O:
            for (s_or_sp, a_tuple), arr in O.items():
                arr = np.asarray(arr, dtype=float)
                # arr may be joint-observation array; index by tuple
                for idx, p in np.ndenumerate(arr):
                    if p:
                        self.observations_compiled[(tuple(int(x) for x in a_tuple), int(s_or_sp), tuple(int(x) for x in idx))] = float(p)
        # compile rewards: parsed.R may have keys (s, a) or (s, a, s')
        self.rewards_compiled = {}
        R = getattr(parsed, "R", None)
        if R:
            for key, val in R.items():
                if isinstance(key, tuple) and len(key) == 3:
                    s, a, sp = key
                    self.rewards_compiled[(int(s), tuple(int(x) for x in a), int(sp))] = float(val)
                elif isinstance(key, tuple) and len(key) == 2:
                    s, a = key
                    # store as (s,a,-1) sentinel for partial key
                    self.rewards_compiled[(int(s), tuple(int(x) for x in a), -1)] = float(val)
                else:
                    # ignore unexpected shapes
                    continue

def _spec_path(name: str) -> str:
    root = os.path.dirname(os.path.dirname(__file__))  # declearn/
    return os.path.join(root, "envs", "masplan_specs", name)

@pytest.mark.parametrize("spec_name", SPEC_NAMES)
def test_sequential_simulator_on_dpomdp(spec_name: str):
    path = _spec_path(spec_name)
    if not os.path.exists(path):
        pytest.skip(f"Spec file missing, skip: {spec_name}")
    parsed = parse_dpomdp(path)
    wrapped = WrappedSpec(parsed)

    sim = SequentialDecPOMDPSimulator(wrapped, seed=0, memory_m=1)

    # reset and record initial obs-history length
    tt = sim.reset()
    assert tt.t == 0
    initial_obs_len = len(sim._obs_history[0])

    # perform one full global time-step (each agent acts once)
    for agent in range(sim.n_agents):
        # choose a valid action 0 (always valid)
        tt, r, done, info = sim.step(0)

    # after the last agent of the time-step, an observation should have been appended
    assert len(sim._obs_history[0]) == initial_obs_len + 1
    # time index should have advanced by one
    assert sim._curr_u.t >= 1
    # simulator should not crash; done is boolean
    assert isinstance(done, bool)