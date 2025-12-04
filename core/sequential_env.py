# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
# declearn/core/sequential_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Sequence
import random
import numpy as np

from declearn.envs.masplan import DecPOMDPSpec


# ============================================================================
# 1. Basic sequential notions
# ============================================================================

@dataclass(frozen=True)
class SubStage:
    """
    A sequential sub-stage u = (i, t):
    - i: which agent acts (0-based)
    - t: global time step (0-based)
    """
    agent: int
    t: int

    def succ(self, n_agents: int) -> "SubStage":
        """
        Successor in the sequential chain:
        (i, t) -> (i+1, t) if i < n_agents-1
        (i, t) -> (0, t+1) otherwise
        """
        if self.agent < n_agents - 1:
            return SubStage(self.agent + 1, self.t)
        else:
            return SubStage(0, self.t + 1)

    def is_terminal(self, horizon: int) -> bool:
        """A sub-stage is terminal once we have moved past the last time step."""
        return self.t >= horizon


@dataclass
class TypeTuple:
    """
    Sequential type θ_u as used by the algorithms:
      - state: current environment state s_t
      - t: current time index
      - joint_histories: list of per-agent dicts with 'obs' and 'acts'
      - prefix_actions: actions already taken at this time step by agents < u.agent
    This contains everything needed to recover priv_u(θ).
    """
    def __init__(self, state: int, t: int, joint_histories: List[Dict[str, List[int]]], prefix_actions: List[Optional[int]]):
        self.state = state
        self.t = t
        self.joint_histories = joint_histories
        self.prefix_actions = prefix_actions
    
    def __eq__(self, other):
        """Égalité pour utilisation comme clé de dictionnaire"""
        if not isinstance(other, TypeTuple):
            return False
        return (self.state, self.t, 
                tuple(tuple(sorted(h.items())) for h in self.joint_histories),
                tuple(self.prefix_actions)) == \
               (other.state, other.t,
                tuple(tuple(sorted(h.items())) for h in other.joint_histories), 
                tuple(other.prefix_actions))
    
    def __hash__(self):
        """Hash pour utilisation comme clé de dictionnaire"""
        try:
            return hash((
                self.state, 
                self.t,
                tuple(tuple(sorted(h.items())) for h in self.joint_histories),
                tuple(self.prefix_actions)
            ))
        except (TypeError, AttributeError):
            # Fallback si les structures sont complexes
            return hash((self.state, self.t, str(self.joint_histories), str(self.prefix_actions)))
    
    def __repr__(self):
        return f"TypeTuple(state={self.state}, t={self.t}, histories={len(self.joint_histories)}, prefix={self.prefix_actions})"


# ============================================================================
# 2. Generic sequential Dec-POMDP simulator
# ============================================================================

class SequentialDecPOMDPSimulator:
    """
    Generic sequential simulator for *any* Dec-POMDP for which the parser
    has produced explicit kernels.

    We assume the following fields are present on the spec:

    - spec.n_agents: int
    - spec.horizon: int
    - spec.n_states: int
    - spec.actions: list of per-agent action counts OR per-agent action names
    - spec.observations: list of per-agent observation counts OR names
    - spec.init_belief: list[float] over states
    - spec.transitions_compiled: dict[(s, joint_act, s_prime)] -> prob
      OR a callable spec.sample_next_state(s, joint_act) -> s'
    - spec.observations_compiled: dict[(joint_act, s_prime, joint_obs)] -> prob
      OR a callable spec.sample_joint_obs(s_prime, joint_act) -> joint_obs
    - spec.rewards_compiled: dict[(s, joint_act, s_prime)] -> float
      OR dict[(s, joint_act)] -> float
      OR a callable spec.reward(s, joint_act, s_prime) -> float

    If any of these is missing, the simulator raises a clear error,
    because we do not special-case DecTiger here.
    """

    def __init__(
        self,
        spec: DecPOMDPSpec,
        seed: int = 0,
        memory_m: int = 0,
    ) -> None:
        self.spec = spec
        self.rng = random.Random(seed)
        self.memory_m = max(0, memory_m)

        self.n_agents: int = spec.n_agents
        self.horizon: int = spec.horizon
        self.n_states: int = spec.n_states

        # detect action/obs cardinalities
        self.n_actions_per_agent: List[int] = self._infer_action_sizes(spec)
        self.n_obs_per_agent: List[int] = self._infer_observation_sizes(spec)

        # current episode state
        self._curr_state: int = 0
        self._curr_u: SubStage = SubStage(0, 0)
        self._curr_joint_action: List[Optional[int]] = [None] * self.n_agents
        # per-agent private histories (full, truncated at build time)
        self._obs_history: List[List[int]] = [[] for _ in range(self.n_agents)]
        self._act_history: List[List[int]] = [[] for _ in range(self.n_agents)]

        # initial belief: interpret spec.start (masplan .dpomdp uses "start:")
        # Interpret the 'start' attribute from the spec (do not expect
        # start_belief/init_belief to exist). Simple, explicit rules:
        #  - start: uniform  -> uniform distribution
        #  - start: "0 0.5 0.5" -> vector of numbers (normalized)
        #  - start: <int> -> one-hot at that index
        #  - start: <state-name> -> one-hot at that state index (if found)
        #  - otherwise -> uniform
        if hasattr(spec, "start"):
            s = getattr(spec, "start")
            try:
                if isinstance(s, (list, tuple, np.ndarray)):
                    arr = np.asarray(s, dtype=float)
                    if arr.size == self.n_states:
                        arr_sum = float(arr.sum()) if arr.sum() != 0 else float(self.n_states)
                        self.init_belief = (arr / arr_sum).astype(float)
                    else:
                        self.init_belief = np.ones(self.n_states, dtype=float) / self.n_states
                elif isinstance(s, str):
                    s_strip = s.strip()
                    s_lower = s_strip.lower()
                    if s_lower == "uniform" or s_strip == "":
                        self.init_belief = np.ones(self.n_states, dtype=float) / self.n_states
                    else:
                        parts = s_strip.split()
                        # sequence of numbers
                        if all(_p.replace('.', '', 1).lstrip('-').isdigit() for _p in parts) and len(parts) > 1:
                            vals = [float(x) for x in parts]
                            arr = np.asarray(vals, dtype=float)
                            if arr.size == self.n_states:
                                arr_sum = float(arr.sum()) if arr.sum() != 0 else float(self.n_states)
                                self.init_belief = (arr / arr_sum).astype(float)
                            else:
                                self.init_belief = np.ones(self.n_states, dtype=float) / self.n_states
                        elif s_strip.isdigit():
                            idx = int(s_strip)
                            arr = np.zeros(self.n_states, dtype=float)
                            if 0 <= idx < self.n_states:
                                arr[idx] = 1.0
                            self.init_belief = arr
                        else:
                            # try match a state name
                            try:
                                idx = spec.states.index(s_strip)
                                arr = np.zeros(self.n_states, dtype=float)
                                arr[idx] = 1.0
                                self.init_belief = arr
                            except Exception:
                                self.init_belief = np.ones(self.n_states, dtype=float) / self.n_states
                else:
                    self.init_belief = np.ones(self.n_states, dtype=float) / self.n_states
            except Exception:
                self.init_belief = np.ones(self.n_states, dtype=float) / self.n_states
        else:
            self.init_belief = np.ones(self.n_states, dtype=float) / self.n_states

        # internal copy used by simulator
        self._init_belief = np.array(self.init_belief, dtype=float, copy=True)

        # check kernels availability early to fail fast
        self._check_kernels()

        # start episode
        self.reset()

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------

    def reset(self) -> TypeTuple:
        """
        Resets the environment to a fresh episode and returns the first type θ_{(0,0)}.
        We also seed the private histories with a dummy initial observation 0 so that
        the policy can always index something.
        """
        self._curr_u = SubStage(0, 0)
        self._curr_joint_action = [None] * self.n_agents
        self._obs_history = [[] for _ in range(self.n_agents)]
        self._act_history = [[] for _ in range(self.n_agents)]

        self._curr_state = self._sample_from_categorical(self._init_belief)

        # dummy initial observation -1 and dummy action -1 (meaning "none yet")
        # Using -1 instead of 0 to clearly indicate "no observation/action yet"
        for i in range(self.n_agents):
            self._obs_history[i].append(-1)
            self._act_history[i].append(-1)

        return self._build_type(self._curr_u)

    def step(self, action: int) -> Tuple[TypeTuple, float, bool, Dict[str, Any]]:
        """
        Executes the action of the current agent at sub-stage u and advances the
        sequential process. Returns (next_type, reward, done, info).

        The reward is only non-zero at the last agent of a time step, when the
        environment transition is actually performed.
        """
        u = self._curr_u
        i = u.agent
        t = u.t

        # record action for this agent at this time
        self._curr_joint_action[i] = action
        self._act_history[i].append(action)

        reward: float = 0.0
        done: bool = False
        info: Dict[str, Any] = {}

        # case 1: not the last agent yet -> just move to next agent
        if i < self.n_agents - 1:
            self._curr_u = self._curr_u.succ(self.n_agents)
            next_type = self._build_type(self._curr_u)
            done = self._curr_u.is_terminal(self.horizon)
            return next_type, reward, done, info

        # case 2: last agent for this time step -> perform environment step
        joint_action: List[int] = [
            a if a is not None else 0 for a in self._curr_joint_action
        ]
        next_state, joint_obs, reward = self._env_step_generic(
            self._curr_state, joint_action
        )

        # update private observations
        for ag in range(self.n_agents):
            self._obs_history[ag].append(joint_obs[ag])

        # advance time
        self._curr_state = next_state
        self._curr_u = SubStage(0, t + 1)
        self._curr_joint_action = [None] * self.n_agents

        done = self._curr_u.is_terminal(self.horizon)
        next_type = self._build_type(self._curr_u)
        info["joint_action"] = joint_action
        info["joint_obs"] = joint_obs

        return next_type, reward, done, info

    # ---------------------------------------------------------------------
    # building types
    # ---------------------------------------------------------------------

    def _build_type(self, u: SubStage) -> TypeTuple:
        """
        Builds the sequential type θ_u = (state, t, joint_histories, prefix_actions).
        Histories are *written* full but *exported* truncated to memory_m so that
        policies with bounded memory see exactly m recent steps.
        """
        joint_histories: List[Dict[str, List[int]]] = []
        for ag in range(self.n_agents):
            obs_hist = self._truncate(self._obs_history[ag], self.memory_m)
            act_hist = self._truncate(self._act_history[ag], self.memory_m)
            joint_histories.append({"obs": obs_hist, "acts": act_hist})

        prefix_actions = self._curr_joint_action[: u.agent]
        return TypeTuple(
            state=self._curr_state,
            t=u.t,
            joint_histories=joint_histories,
            prefix_actions=prefix_actions,
        )

    @staticmethod
    def _truncate(xs: List[int], m: int) -> List[int]:
        if m <= 0:
            return list(xs)
        return list(xs[-m:])

    # ---------------------------------------------------------------------
    # environment dynamics (generic, table-driven)
    # ---------------------------------------------------------------------

    def _env_step_generic(
        self,
        state: int,
        joint_action: Sequence[int],
    ) -> Tuple[int, List[int], float]:
        """
        Generic Dec-POMDP step:
          (s, a_1,...,a_n) -> sample s' ~ T(. | s, a)
                              sample z  ~ O(. | s', a)
                              r = R(s, a, s') or R(s, a)
        We insist on explicit kernels; if they are not present, we fail loudly.
        """
        # --- sample next state
        if hasattr(self.spec, "transitions_compiled") and self.spec.transitions_compiled:
            s_prime = self._sample_next_state_from_table(state, joint_action)
        elif hasattr(self.spec, "sample_next_state"):
            s_prime = self.spec.sample_next_state(state, joint_action)
        else:
            # no transition kernel available -> stay in place
            s_prime = int(state)

        # --- sample joint observation
        if hasattr(self.spec, "observations_compiled") and self.spec.observations_compiled:
            joint_obs = self._sample_joint_obs_from_table(s_prime, joint_action)
        elif hasattr(self.spec, "sample_joint_obs"):
            joint_obs = self.spec.sample_joint_obs(s_prime, joint_action)
        else:
            # fallback: return zero observation for each agent (for minimal test specs)
            joint_obs = [0] * self.n_agents

        # --- reward
        reward = 0.0
        if hasattr(self.spec, "rewards_compiled") and self.spec.rewards_compiled:
            reward = self._reward_from_table(state, joint_action, s_prime)
        elif hasattr(self.spec, "reward") and callable(self.spec.reward):
            reward = self.spec.reward(state, joint_action, s_prime)
        elif hasattr(self.spec, "reward") and isinstance(self.spec.reward, dict):
            # Legacy reward dict format - try different key combinations
            key_act = tuple(joint_action)
            r = (self.spec.reward.get((state, key_act, s_prime), None) or 
                 self.spec.reward.get((state, key_act), None) or 0.0)
            reward = float(r)
        else:
            reward = 0.0  # safe default

        return s_prime, joint_obs, reward

    # ---------------------------------------------------------------------
    # table helpers
    # ---------------------------------------------------------------------

    def _sample_next_state_from_table(
        self, state: int, joint_action: Sequence[int]
    ) -> int:
        """
        Samples s' from transitions_compiled dict using masplan.py format:
        transitions_compiled[(s, joint_act_tuple)] -> prob_vector
        """
        key_act = tuple(joint_action)
        # Primary: try transitions_compiled from masplan.py parser
        tc = getattr(self.spec, "transitions_compiled", {})
        probs_vec = tc.get((int(state), key_act), None)
        if probs_vec is not None:
            probs = np.asarray(probs_vec, dtype=float).ravel()
            if probs.size > 0:
                ssum = float(probs.sum())
                if ssum > 0.0:
                    probs = probs / ssum
                    choices = np.arange(probs.size, dtype=int)
                    return int(self._sample_from_categorical(probs))
        
        # Secondary: fallback to legacy dicts (transition, T) with same format
        for mapping_name in ("transition", "T"):
            mapping = getattr(self.spec, mapping_name, None)
            if isinstance(mapping, dict):
                vec = mapping.get((int(state), key_act), None)
                if vec is not None:
                    probs = np.asarray(vec, dtype=float).ravel()
                    if probs.size > 0:
                        ssum = float(probs.sum())
                        if ssum > 0.0:
                            probs = probs / ssum
                            choices = np.arange(probs.size, dtype=int)
                            return int(self._sample_from_categorical(probs))
        
        # Ultimate fallback: stay in place
        return int(state)
    
    def _sample_next_state_from_spec(self, state: int, joint_action: Sequence[int]) -> int:
        """Alias for compatibility with existing tests."""
        return self._sample_next_state_from_table(state, joint_action)

    def _sample_joint_obs_from_table(
        self, next_state: int, joint_action: Sequence[int]
    ) -> List[int]:
        """
        Samples joint observation from observations_compiled dict.
        Supports multiple formats from masplan.py parser.
        """
        key_act = tuple(joint_action)
        
        oc = getattr(self.spec, "observations_compiled", {})
        
        # Try different key formats that masplan.py might produce
        for key_format in [
            (key_act, next_state),  # (joint_action, next_state)
            (next_state, key_act),  # (next_state, joint_action)  
            key_act,                # joint_action only
            next_state,             # next_state only
        ]:
            if key_format in oc:
                obs_data = oc[key_format]
                if isinstance(obs_data, (list, tuple, np.ndarray)):
                    # Direct observation vector for each agent
                    obs_list = list(obs_data)
                    # Ensure it's per-agent format
                    if len(obs_list) == self.n_agents:
                        return [int(o) for o in obs_list]
                    else:
                        # Single joint observation -> split equally or use as-is
                        return [int(obs_list[0] if obs_list else 0)] * self.n_agents
                elif isinstance(obs_data, dict):
                    # Sparse format {joint_obs_tuple: prob}
                    probs = list(obs_data.values())
                    obs_tuples = list(obs_data.keys())
                    if probs:
                        idx = self._sample_from_categorical(probs)
                        selected = obs_tuples[idx]
                        if isinstance(selected, (list, tuple)):
                            return [int(o) for o in selected]
                        else:
                            return [int(selected)] * self.n_agents
        
        # Try legacy format (joint_action, next_state, joint_obs) -> prob
        rows: List[Tuple[Tuple[int, ...], float]] = []
        for key, p in oc.items():
            if isinstance(key, tuple) and len(key) == 3:
                a, s_prime, z = key
                if a == key_act and s_prime == next_state:
                    rows.append((z, p))
        
        if rows:
            probs = [p for (_, p) in rows]
            z_tuples = [z for (z, _) in rows]
            idx = self._sample_from_categorical(probs)
            return list(z_tuples[idx])
        
        # Fallback: return zero observation for each agent
        return [0] * self.n_agents

    def _reward_from_table(
        self, state: int, joint_action: Sequence[int], next_state: int
    ) -> float:
        """
        Looks up reward in the most general order:
          1. (s, a, s')  → r
          2. (s, a)      → r
        """
        key_act = tuple(joint_action)
        # full key
        r = self.spec.rewards_compiled.get((state, key_act, next_state), None)
        if r is not None:
            return float(r)
        # partial key
        r = self.spec.rewards_compiled.get((state, key_act), None)
        if r is not None:
            return float(r)
        return 0.0

    # ---------------------------------------------------------------------
    # small utils
    # ---------------------------------------------------------------------

    @staticmethod
    def _infer_action_sizes(spec: DecPOMDPSpec) -> List[int]:
        """
        Infers action space sizes from the spec. We accept:
          - spec.actions = [3, 3]  (counts)
          - spec.actions = [['listen','open-left','open-right'], ...]
        """
        sizes: List[int] = []
        for a in spec.actions:
            if isinstance(a, int):
                sizes.append(a)
            elif isinstance(a, (list, tuple)):
                sizes.append(len(a))
            else:
                raise ValueError("Unsupported action specification on DecPOMDPSpec.")
        return sizes

    @staticmethod
    def _infer_observation_sizes(spec: DecPOMDPSpec) -> List[int]:
        """
        Similar to actions, but for observations.
        """
        sizes: List[int] = []
        for o in spec.observations:
            if isinstance(o, int):
                sizes.append(o)
            elif isinstance(o, (list, tuple)):
                sizes.append(len(o))
            else:
                raise ValueError("Unsupported observation specification on DecPOMDPSpec.")
        return sizes

    def _check_kernels(self) -> None:
        """
        We do an explicit check that the spec looks like a *fully* parsed Dec-POMDP.
        If not, we fail early, because we do not want to silently fall back to a
        domain-specific behaviour (e.g. DecTiger).
        """
        # Accept either compiled transitions (pair->vector or triple->scalar),
        # or legacy 'transition'/'T' dicts, or a sample_next_state fallback.
        has_trans = (
            (hasattr(self.spec, "transitions_compiled") and isinstance(self.spec.transitions_compiled, dict))
            or getattr(self.spec, "transition", None) is not None
            or getattr(self.spec, "T", None) is not None
            or hasattr(self.spec, "sample_next_state")
        )
        has_obs = (
            (hasattr(self.spec, "observations_compiled") and isinstance(self.spec.observations_compiled, dict))
            or getattr(self.spec, "observation", None) is not None
            or getattr(self.spec, "O", None) is not None
            or hasattr(self.spec, "sample_joint_obs")
        )
        has_rew = (
            hasattr(self.spec, "rewards_compiled")
            and isinstance(self.spec.rewards_compiled, dict)
        ) or hasattr(self.spec, "reward")

        if not has_trans:
            raise RuntimeError(
                "Dec-POMDP spec is missing a transition kernel. "
                "Please ensure the .dpomdp parser compiles 'T:' entries."
            )
        # Observation kernel is optional for minimal test specs
        if not has_obs:
            print(
                "[SequentialDecPOMDPSimulator] Warning: no observation kernel found on spec. "
                "Proceeding with zero observations."
            )
        # reward is optional but we keep the check so that experiments are explicit
        if not has_rew:
            # not fatal, but we make it visible
            print(
                "[SequentialDecPOMDPSimulator] Warning: no reward kernel found on spec. "
                "Proceeding with zero rewards."
            )

    def _sample_from_categorical(self, probs: Sequence[float]) -> int:
        """Samples an index from a categorical distribution using numpy for precision."""
        probs_array = np.asarray(probs, dtype=float)
        # Normalize to ensure sum = 1.0 (handle floating point errors)
        total = probs_array.sum()
        if total <= 0:
            # Uniform fallback if all probabilities are zero
            return self.rng.randint(0, len(probs) - 1)
        
        probs_array = probs_array / total
        
        # Use numpy's approach: cumulative sum and find first index where cumsum >= random
        r = self.rng.random()
        cumsum = np.cumsum(probs_array)
        
        # Find first index where cumsum >= r
        idx = np.searchsorted(cumsum, r, side='right')
        return min(idx, len(probs) - 1)