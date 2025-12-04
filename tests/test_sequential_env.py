# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import os
import itertools
import pytest
from dataclasses import dataclass
from typing import Tuple, Dict
from declearn.envs.masplan import parse_dpomdp
from declearn.core.sequential_env import SequentialDecPOMDPSimulator

@dataclass
class FakeSpec:
    n_agents: int
    horizon: int
    n_states: int
    actions: list
    observations: list
    init_belief: list
    transitions_compiled: Dict[Tuple[int, Tuple[int, ...], int], float]
    observations_compiled: Dict[Tuple[Tuple[int, ...], int, Tuple[int, ...]], float]
    rewards_compiled: Dict[Tuple[int, Tuple[int, ...]], float]

def make_simple_spec():
    n_agents = 2
    horizon = 3
    n_states = 2
    actions = [2, 2]        # each agent has 2 actions: 0,1
    observations = [2, 2]   # each agent has 2 observations: 0,1
    init_belief = [1.0, 0.0]  # start in state 0 deterministically

    transitions = {}
    observations_map = {}
    rewards = {}

    # build deterministic transitions/observations: s' = s (stay),
    # obs = joint action tuple (clamped by obs size), reward = sum(actions)
    for s in range(n_states):
        for a in itertools.product(*[range(x) for x in actions]):
            # deterministic stay
            transitions[(s, tuple(a), s)] = 1.0
            # observation deterministic equal to action (mod observation sizes)
            z = tuple((ai % obs) for ai, obs in zip(a, observations))
            observations_map[(tuple(a), s, z)] = 1.0
            # reward keyed by (s, a) -> sum of action indices
            rewards[(s, tuple(a))] = float(sum(a))

    return FakeSpec(
        n_agents=n_agents,
        horizon=horizon,
        n_states=n_states,
        actions=actions,
        observations=observations,
        init_belief=init_belief,
        transitions_compiled=transitions,
        observations_compiled=observations_map,
        rewards_compiled=rewards,
    )

def test_sequential_simulator_sequence_and_rewards():
    spec = make_simple_spec()
    sim = SequentialDecPOMDPSimulator(spec, seed=0, memory_m=2)

    # reset -> initial type
    tt = sim.reset()
    assert tt.t == 0
    assert tt.state in range(spec.n_states)
    assert tt.prefix_actions == []  # no prefix at first substage
    # first agent acts (not last) -> reward 0, prefix_actions populated
    tt, r, done, info = sim.step(1)
    assert r == 0.0
    assert not done
    # after first agent acted, prefix_actions should contain that action
    assert tt.prefix_actions == [1]
    # second agent acts (last agent) -> environment step occurs
    tt, r, done, info = sim.step(0)
    # reward should equal sum of joint action (1+0)=1.0 as defined
    assert r == 1.0
    assert "joint_action" in info and isinstance(info["joint_action"], list)
    assert "joint_obs" in info and isinstance(info["joint_obs"], list)
    # observation history must have been appended
    assert sim._obs_history[0][-1] == info["joint_obs"][0]
    assert sim._obs_history[1][-1] == info["joint_obs"][1]

    # run until horizon reached; simple policy: agents always choose action 0
    steps = 0
    while not done:
        agent_idx = len(tt.prefix_actions)
        act = 0
        tt, r, done, info = sim.step(act)
        steps += 1
        # rewards are floats
        assert isinstance(r, float)

    # after termination, substage t should be >= horizon
    assert sim._curr_u.t >= spec.horizon
    # total number of step calls should not exceed horizon * n_agents + a small margin
    assert steps <= spec.horizon * spec.n_agents + 2

def _specs_dir() -> str:
    root = os.path.dirname(os.path.dirname(__file__))  # declearn/
    return os.path.join(root, "envs", "masplan_specs")

def test_scan_all_dpomdp_specs():
    specs_dir = _specs_dir()
    if not os.path.isdir(specs_dir):
        pytest.skip("Aucun répertoire masplan_specs trouvé")
    dpomdp_files = [os.path.join(specs_dir, f) for f in os.listdir(specs_dir) if f.endswith(".dpomdp")]
    if not dpomdp_files:
        pytest.skip("Aucun fichier .dpomdp à tester")

    for path in sorted(dpomdp_files):
        name = os.path.basename(path)
        # parsing : doit réussir
        try:
            spec = parse_dpomdp(path)
        except Exception as e:
            pytest.fail(f"Échec du parsing pour {name}: {e}")

        # simulation minimale : instanciation + reset + un pas global
        try:
            sim = SequentialDecPOMDPSimulator(spec, seed=0, memory_m=1)
            tt = sim.reset()
            # effectuer au plus une étape globale : chaque agent joue une action (ici 0)
            for _ in range(sim.n_agents):
                tt, r, done, info = sim.step(0)
            # vérifications basiques
            assert isinstance(done, bool)
            assert isinstance(r, (float, int))
        except Exception as e:
            pytest.fail(f"Échec de la simulation pour {name}: {e}")

# compatibilité nommer parse également si d'autres modules l'attendent
parse = parse_dpomdp  # type: ignore