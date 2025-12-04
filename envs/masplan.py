# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
# declearn/envs/masplan.py
import re
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import os
import sys
import numpy as np
from itertools import product
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class DecPOMDPSpec:
    n_agents: int = 0
    states: List[str] = field(default_factory=list)
    actions: List[List[str]] = field(default_factory=list)
    observations: List[List[str]] = field(default_factory=list)

    # canonical start (parser fournit la croyance initiale normalisée ici)
    start: Optional[Any] = None
    start_belief: Optional[np.ndarray] = None
    init_belief: Optional[np.ndarray] = None

    # dynamics / observation / reward kernels (parser fills these)
    transition: Dict = field(default_factory=dict)
    observation: Dict = field(default_factory=dict)
    reward: Dict = field(default_factory=dict)

    # metadata / misc parser fields
    discount: float = 1.0
    values: str = "reward"
    source_path: Optional[str] = None
    horizon: int = 1
    n_states: Optional[int] = None

    # optional compiled representations (populated by parser if available)
    transitions_compiled: Optional[Dict] = None
    observations_compiled: Optional[Dict] = None
    rewards_compiled: Optional[Dict] = None

    # helpers that parser/simulator may attach later
    n_actions_per_agent: Optional[List[int]] = None
    n_obs_per_agent: Optional[List[int]] = None

    def __post_init__(self):
        if self.n_states is None:
            self.n_states = len(self.states)

    def summary(self) -> str:
        """Retourne une courte description textuelle du spec (utilisée dans les tests)."""
        try:
            na = int(getattr(self, "n_agents", 0))
            ns = int(getattr(self, "n_states", len(getattr(self, "states", []))))
        except Exception:
            na = getattr(self, "n_agents", 0)
            ns = len(getattr(self, "states", []))
        acts = [len(a) for a in getattr(self, "actions", [])] if getattr(self, "actions", None) else []
        obs = [len(o) for o in getattr(self, "observations", [])] if getattr(self, "observations", None) else []
        disc = getattr(self, "discount", None)
        hor = getattr(self, "horizon", None)
        parts = [
            f"agents={na}",
            f"states={ns}",
            f"actions={acts}",
            f"observations={obs}",
            f"discount={disc}",
            f"horizon={hor}",
        ]
        return ", ".join(str(p) for p in parts)


def _split_tokens(line: str) -> List[str]:
    return [t for t in line.strip().split() if t]


def _validate_spec(spec, path: Optional[str] = None) -> None:
    """Valide qu'un DecPOMDPSpec ne contient aucun attribut critique à None.
    Lève ValueError si une validation échoue.

    NOTE: `horizon` est facultatif pour certains .dpomdp; on le gère ensuite via une
    valeur par défaut si nécessaire.
    """
    required = {
        "n_agents": lambda v: isinstance(v, int) and v > 0,
        "states": lambda v: isinstance(v, list) and len(v) > 0,
        "actions": lambda v: isinstance(v, list) and len(v) > 0,
        "observations": lambda v: isinstance(v, list) and len(v) > 0,
        "start": lambda v: v is not None,
        "transition": lambda v: isinstance(v, dict),
        "observation": lambda v: isinstance(v, dict),
        "reward": lambda v: isinstance(v, dict),
        # horizon intentionally not required here (some specs omit it)
    }
    missing = []
    bad = []
    for name, check in required.items():
        if not hasattr(spec, name):
            missing.append(name)
        else:
            val = getattr(spec, name)
            if val is None or not check(val):
                bad.append(name)
    if missing or bad:
        ctx = f" for {path}" if path else ""
        msgs = []
        if missing:
            msgs.append(f"missing attrs: {missing}")
        if bad:
            msgs.append(f"invalid/None attrs: {bad}")
        raise ValueError(f"Invalid DecPOMDPSpec{ctx}: " + "; ".join(msgs))


def parse(path: str) -> DecPOMDPSpec:
    """
    Try new parser first; new parser MUST return a valid DecPOMDPSpec (never None).
    If it fails, fallback to legacy. If fallback also fails, raise with clear message.
    """
    # try import new parser
    try:
        from declearn.parsers.dpomdp_parser import parse as new_parse
    except Exception:
        return _legacy_parse_dpomdp(path)

    # call new parser
    try:
        result = new_parse(path)
    except Exception as e:
        # new parser raised -> fallback to legacy
        return _legacy_parse_dpomdp(path)

    # if new parser returned None or invalid, raise so we can fix the parser
    if result is None:
        raise RuntimeError(f"New parser declearn.parsers.dpomdp_parser.parse returned None for {path!r}")
    if not hasattr(result, "n_agents"):
        raise RuntimeError(f"New parser returned object without 'n_agents' for {path!r}: {type(result)}")

    # final validation
    _validate_spec(result, path)
    return result


def _legacy_parse_dpomdp(path: str) -> DecPOMDPSpec:
    """
    Parser for MASPlan / Cassandra-like .dpomdp files.
    We support the common pattern:

      agents: 2
      discount: 0.95
      values: reward
      states: s1 s2 ...
      actions: a1 a2 ...          (or block: actions: \\n a1 a2 \\n a1 a2)
      observations: o1 o2 ...     (or block: observations: \\n o1 o2 \\n o1 o2)
      start: 0.5 0.5 ...
      T: * a1 a2 : s1 : 0.7 0.3   etc.
      O: * a1 a2 : o1 o2 : 0.5 ...
      R: s1 a1 a2 : * : 10

    We normalise to indexes.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and not l.strip().startswith("#")]

    n_agents = 1
    discount = 1.0
    values = "reward"
    states: List[str] = []
    actions_raw: List[str] = []
    observations_raw: List[str] = []
    horizon: Optional[int] = None
    start_belief: Optional[np.ndarray] = None

    # interim storages
    T: Dict[Tuple[int, Tuple[int, ...]], np.ndarray] = {}
    O: Dict[Tuple[int, Tuple[int, ...]], np.ndarray] = {}
    R: Dict[Tuple[int, Tuple[int, ...]], float] = {}

    debug = os.environ.get("MASPLAN_DEBUG") == "1"
    # first pass: headers (support single-line or block-style entries; '|' variant removed)
    i = 0
    while i < len(lines):
        line = lines[i]
        if debug and (i % 50 == 0):
            print(f"[masplan] first-pass i={i} line={line!r}", file=sys.stderr)
        low = line.lower()
        if low.startswith("agents:"):
            n_agents = int(line.split(":", 1)[1].strip())
            i += 1
            continue
        elif low.startswith("discount:"):
            discount = float(line.split(":", 1)[1].strip())
            i += 1
            continue
        elif low.startswith("values:"):
            values = line.split(":", 1)[1].strip()
            i += 1
            continue
        elif low.startswith("states:"):
            states = line.split(":", 1)[1].strip().split()
            i += 1
            continue
        elif low.startswith("actions:"):
            acts = line.split(":", 1)[1].strip()
            # block form: following lines (one per agent) until next header (line containing ':')
            if acts == "":
                collected = []
                j = i + 1
                while j < len(lines) and ":" not in lines[j]:
                    collected.append(lines[j])
                    j += 1
                if collected:
                    # each collected line is actions for one agent
                    actions_raw = [c.split() for c in collected]
                    i = j
                    continue
                else:
                    actions_raw = []
                    i += 1
                    continue
            else:
                # single-line: same action set duplicated for all agents
                actions_raw = acts.split()
                i += 1
                continue
        elif low.startswith("observations:"):
            obs = line.split(":", 1)[1].strip()
            if obs == "":
                collected = []
                j = i + 1
                while j < len(lines) and ":" not in lines[j]:
                    collected.append(lines[j])
                    j += 1
                if collected:
                    observations_raw = [c.split() for c in collected]
                    i = j
                    continue
                else:
                    observations_raw = []
                    i += 1
                    continue
            else:
                observations_raw = obs.split()
                i += 1
                continue
        elif low.startswith("start:"):
            rest = line.split(":", 1)[1].strip()
            if rest == "" and i + 1 < len(lines) and ":" not in lines[i + 1]:
                rest = lines[i + 1].strip()
                i += 1
            parts = rest.split() if rest else []
            try:
                if parts:
                    start_belief = np.array([float(x) for x in parts], dtype=np.float32)
            except ValueError:
                start_belief = None
            i += 1
            continue
        elif low.startswith("horizon:"):
            horizon = int(line.split(":", 1)[1].strip())
            i += 1
            continue
        else:
            i += 1
            continue

    # validations
    if not actions_raw:
        raise ValueError(f"No 'actions' section parsed in dpomdp file: {path}")
    if not observations_raw:
        raise ValueError(f"No 'observations' section parsed in dpomdp file: {path}")

    # normalize to per-agent lists
    if isinstance(actions_raw[0], list):
        actions: List[List[str]] = actions_raw  # already per agent
    else:
        actions = [actions_raw for _ in range(n_agents)]
    if isinstance(observations_raw[0], list):
        observations: List[List[str]] = observations_raw  # already per agent
    else:
        observations = [observations_raw for _ in range(n_agents)]

    # allow "states: N" (single number) -> expand to state names "0..N-1"
    if len(states) == 1 and states[0].isdigit():
        n_st = int(states[0])
        states = [str(i) for i in range(n_st)]

    # allow per-agent observation counts like "2" -> expand to ["0","1"]
    for ag in range(len(observations)):
        obs_list = observations[ag]
        if len(obs_list) == 1 and obs_list[0].isdigit():
            m = int(obs_list[0])
            observations[ag] = [str(i) for i in range(m)]

    # similarly for actions: if a single numeric token, interpret as count
    for ag in range(len(actions)):
        act_list = actions[ag]
        if len(act_list) == 1 and act_list[0].isdigit():
            m = int(act_list[0])
            actions[ag] = [str(i) for i in range(m)]

    n_states = len(states)
    if start_belief is None:
        start_belief = np.ones(n_states, dtype=np.float32) / n_states

    # maps from name -> idx
    s2i = {s: i for i, s in enumerate(states)}
    a2i = [{a: i for i, a in enumerate(actions[ag])} for ag in range(n_agents)]
    z2i = [{z: i for i, z in enumerate(observations[ag])} for ag in range(n_agents)]

    # helper to parse joint action
    def parse_joint_action(tokens: List[str]) -> Tuple[int, ...]:
        # tokens = ['a1', 'a2'] per agent
        if len(tokens) != n_agents:
            raise ValueError(f"Joint-action length mismatch: {tokens} vs n_agents={n_agents}")
        out = []
        for ag, tok in enumerate(tokens):
            if isinstance(tok, int) or (isinstance(tok, str) and tok.isdigit()):
                out.append(int(tok))
            else:
                out.append(a2i[ag][tok])
        return tuple(out)

    # second pass: T: O: R: (support single-line and block "uniform"/"identity" forms)
    # precompute all joint actions
    all_joints = list(product(*[range(len(actions[ag])) for ag in range(n_agents)]))
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if debug and (idx % 50 == 0):
            print(f"[masplan] second-pass idx={idx} line={line!r}", file=sys.stderr)
        if line.startswith("T:"):
            parts = [p.strip() for p in line[2:].strip().split(":")]
            # try to detect which part is actions / start / next-state among the first 3 parts
            if len(parts) >= 4:
                prob_tokens = parts[3].split()

                def is_action_tokens(tok_str: str) -> bool:
                    toks = tok_str.split()
                    if not toks:
                        return False
                    if toks[0] == "*":
                        return True
                    if len(toks) != n_agents:
                        return False
                    for ag, t in enumerate(toks):
                        # accept numeric indices or named actions
                        if t.isdigit():
                            # index must be within range
                            idx = int(t)
                            if idx < 0 or idx >= len(actions[ag]):
                                return False
                            continue
                        if t not in a2i[ag]:
                            return False
                    return True

                def is_state_token(tok_str: str) -> bool:
                    return tok_str == "*" or tok_str in s2i

                # detect action index among first three parts
                idx_action = None
                for k in range(min(3, len(parts))):
                    if is_action_tokens(parts[k]):
                        idx_action = k
                        break

                # default ordering fallback
                if idx_action is None:
                    # assume conventional: start-state, actions, next-state
                    s_from_part = parts[0]
                    a_tokens = parts[1].split()
                    s_to_part = parts[2]
                else:
                    # derive start / next-state from remaining slots
                    rem = [k for k in range(min(3, len(parts))) if k != idx_action]
                    if len(rem) >= 2 and is_state_token(parts[rem[0]]) and is_state_token(parts[rem[1]]):
                        s_from_part = parts[rem[0]]
                        s_to_part = parts[rem[1]]
                    else:
                        if is_state_token(parts[rem[0]]):
                            s_from_part = parts[rem[0]]
                            s_to_part = parts[rem[1]] if len(rem) > 1 else "*"
                        else:
                            s_from_part = "*"
                            s_to_part = parts[rem[0]]
                    a_tokens = parts[idx_action].split()

                probs = np.array([float(x) for x in prob_tokens], dtype=np.float32)

                # wildcard actions '*' -> apply to all joint actions
                if a_tokens and a_tokens[0] == "*":
                    if s_from_part == "*":
                        for si in range(n_states):
                            for a_joint in all_joints:
                                T[(si, a_joint)] = probs
                    else:
                        si = s2i[s_from_part]
                        for a_joint in all_joints:
                            T[(si, a_joint)] = probs
                else:
                    if len(a_tokens) != n_agents:
                        raise ValueError(f"Invalid joint-action token count in line: {line!r}")
                    a_joint = parse_joint_action(a_tokens)
                    if s_from_part == "*":
                        for si in range(n_states):
                            T[(si, a_joint)] = probs
                    else:
                        si = s2i[s_from_part]
                        T[(si, a_joint)] = probs
            idx += 1
            continue

        elif line.startswith("O:"):
            parts = [p.strip() for p in line[2:].strip().split(":")]
            if len(parts) >= 4:
                # parts may appear in different orders for O:
                # known forms:
                # 1) "<start-state> : <actions> : <obs> : <probs>"
                # 2) "<actions> : <end-state> : <obs> : <probs>"
                prob_part = parts[3].split()

                def is_action_tokens(tok_list: str) -> bool:
                    toks = tok_list.split()
                    if not toks:
                        return False
                    if toks[0] == "*":
                        return True
                    if len(toks) != n_agents:
                        return False
                    for ag, t in enumerate(toks):
                        # accept numeric indices or named actions
                        if t.isdigit():
                            # index must be within range
                            idx = int(t)
                            if idx < 0 or idx >= len(actions[ag]):
                                return False
                            continue
                        if t not in a2i[ag]:
                            return False
                    return True

                def is_obs_tokens(tok_list: str) -> bool:
                    toks = tok_list.split()
                    if len(toks) != n_agents:
                        return False
                    for ag, t in enumerate(toks):
                        if t.isdigit():
                            ii = int(t)
                            if ii < 0 or ii >= len(observations[ag]):
                                return False
                            continue
                        if t not in z2i[ag]:
                            return False
                    return True

                # detect indexes
                idx_action = None
                idx_state = None
                idx_obs = None
                # check among the first three parts
                for k in range(3):
                    if idx_action is None and is_action_tokens(parts[k]):
                        idx_action = k
                        continue
                for k in range(3):
                    if idx_obs is None and is_obs_tokens(parts[k]):
                        idx_obs = k
                        continue
                # state is the remaining one (prefer single-token matching a state)
                for k in range(3):
                    if k == idx_action or k == idx_obs:
                        continue
                    toks_k = parts[k].split()
                    if not toks_k:
                        continue
                    # prefer part whose first token is a state or wildcard
                    if toks_k[0] == "*" or toks_k[0] in s2i:
                        idx_state = k
                        break
                # fallback to conventional ordering if detection failed
                if idx_action is None:
                    # assume actions at 1
                    idx_action = 1
                if idx_obs is None:
                    idx_obs = 2
                if idx_state is None:
                    idx_state = 0

                a_tokens = parts[idx_action].split()
                o_part = parts[idx_obs].split()
                s_part = parts[idx_state]

                # target joint actions
                if a_tokens and a_tokens[0] == "*":
                    target_joints = all_joints
                else:
                    if len(a_tokens) != n_agents:
                        raise ValueError(f"Invalid joint-action token count in line: {line!r}")
                    target_joints = [parse_joint_action(a_tokens)]

                if s_part == "*":
                    for si in range(n_states):
                        if len(o_part) == n_agents:
                            # support numeric obs tokens
                            o_joint_idx = []
                            for ag in range(n_agents):
                                tok = o_part[ag]
                                if tok.isdigit():
                                    o_joint_idx.append(int(tok))
                                else:
                                    o_joint_idx.append(z2i[ag][tok])
                            o_joint_idx = tuple(o_joint_idx)
                            joint_shape = tuple(len(observations[ag]) for ag in range(n_agents))
                            arr = np.zeros(joint_shape, dtype=np.float32)
                            arr[o_joint_idx] = float(prob_part[0])
                            for a_joint in target_joints:
                                O[(si, a_joint)] = arr
                else:
                    # support numeric state tokens (index)
                    if s_part.isdigit():
                        si = int(s_part)
                    else:
                        try:
                            si = s2i[s_part]
                        except KeyError:
                            # fallback: if s_part contains multiple tokens, try to recover
                            toks = s_part.split()
                            si = None
                            for t in toks:
                                if t.isdigit():
                                    si = int(t)
                                    break
                            if si is None:
                                # last resort: raise with clearer message
                                raise KeyError(f"Unknown state token in O: {s_part!r} (line: {line!r})")
                    if len(o_part) == n_agents:
                        o_joint_idx = []
                        for ag in range(n_agents):
                            tok = o_part[ag]
                            if tok.isdigit():
                                o_joint_idx.append(int(tok))
                            else:
                                o_joint_idx.append(z2i[ag][tok])
                        o_joint_idx = tuple(o_joint_idx)
                        joint_shape = tuple(len(observations[ag]) for ag in range(n_agents))
                        arr = np.zeros(joint_shape, dtype=np.float32)
                        arr[o_joint_idx] = float(prob_part[0])
                        for a_joint in target_joints:
                            O[(si, a_joint)] = arr
                idx += 1
                continue

            # block forms: e.g. "O: * :" followed by "uniform"
            j = idx + 1
            tail = []
            while j < len(lines) and not lines[j].startswith(("T:", "O:", "R:")):
                tail.append(lines[j])
                j += 1
            if len(tail) == 1 and tail[0] == "uniform":
                prefix = parts[0] if parts else ""
                toks = prefix.split() if prefix else []
                if prefix == "*" or toks == ["*"]:
                    for si in range(n_states):
                        for a_joint in all_joints:
                            joint_shape = tuple(len(observations[ag]) for ag in range(n_agents))
                            O[(si, a_joint)] = np.ones(joint_shape, dtype=np.float32)
                else:
                    if toks and len(toks) == n_agents:
                        try:
                            a_joint = parse_joint_action(toks)
                            joint_shape = tuple(len(observations[ag]) for ag in range(n_agents))
                            for si in range(n_states):
                                O[(si, a_joint)] = np.ones(joint_shape, dtype=np.float32)
                        except KeyError:
                            pass
                idx = j
                continue
            else:
                idx = j
                continue

        elif line.startswith("R:"):
            parts = [p.strip() for p in line[2:].strip().split(":")]
            r_part = parts[-1].split()

            # Format POMDP standard: "R: actions : state : * : reward"
            # ou "R: * : state : * : reward" pour toutes les actions
            if len(parts) >= 4:
                # Format standard: R: actions : state : * : reward
                a_part = parts[0].split()
                s_part = parts[1].strip()
                # parts[2] est ignoré (souvent "*")
                # parts[3] est la récompense
                
                # Traiter les actions
                if len(a_part) == 1 and a_part[0] == "*":
                    # Wildcard simple pour toutes les actions
                    target_joints = all_joints
                elif len(a_part) == n_agents and all(token == "*" for token in a_part):
                    # Wildcards multiples (un par agent) pour toutes les actions
                    target_joints = all_joints
                else:
                    # Actions spécifiques
                    if len(a_part) != n_agents:
                        raise ValueError(f"Expected {n_agents} actions, got {len(a_part)} in line: {line!r}")
                    target_joints = [parse_joint_action(a_part)]
                
                # Traiter l'état
                if s_part == "*":
                    # Wildcard pour tous les états
                    target_states = list(range(n_states))
                else:
                    # État spécifique
                    if s_part.isdigit():
                        si = int(s_part)
                        if si < 0 or si >= n_states:
                            raise ValueError(f"State index out of range: {si} in line: {line!r}")
                        target_states = [si]
                    else:
                        if s_part not in s2i:
                            raise ValueError(f"Unknown state: {s_part} in line: {line!r}")
                        target_states = [s2i[s_part]]
                
                # Assigner les récompenses
                reward_value = float(r_part[0])
                for si in target_states:
                    for a_joint in target_joints:
                        R[(si, a_joint)] = reward_value
            else:
                raise ValueError(f"Invalid reward line format: {line!r}")
            idx += 1
            continue
        else:
            # advance when line is not T:/O:/R:
            idx += 1
            continue
    # L'horizon n'est PAS un attribut du format .dpomdp standard
    # Il doit être fourni à l'exécution, donc on le laisse à None si absent
    # (le simulateur ou l'algorithme devra le spécifier explicitement)
    
    # build and return canonical spec
    spec = DecPOMDPSpec(
        n_agents=n_agents,
        states=states,
        actions=actions,
        observations=observations,
        start=start_belief,
        start_belief=start_belief,
        transition=T,
        observation=O,
        reward=R,
        discount=discount,
        horizon=horizon,
        values=values,
        source_path=path,
    )

    # L'horizon reste None si non spécifié dans le fichier .dpomdp
    # C'est intentionnel : l'horizon est un paramètre d'exécution, pas du fichier

    # validation stricte : aucun attribut critique ne doit être None
    _validate_spec(spec, path)

    # Compatibilité : alias bruts + tables "compilées" attendues par le simulateur
    spec.T = spec.transition
    spec.O = spec.observation
    spec.R = spec.reward
    # spec.start contient la croyance initiale normalisée fournie par le parser
    spec.init_belief = spec.start

    # transitions_compiled : map (s, a_tuple) -> prob vector
    transitions_compiled = {}
    nstates_local = int(getattr(spec, "n_states", len(getattr(spec, "states", []))))
    for (s_key, a_key), probs in spec.transition.items():
        try:
            probs_arr = np.asarray(probs, dtype=float).ravel()
        except Exception:
            probs_arr = np.ones(nstates_local, dtype=float) / float(max(1, nstates_local))
        # if length mismatch, try sparse (s', p) pairs or fallback to uniform
        if probs_arr.size != nstates_local:
            if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
                vec = np.zeros(nstates_local, dtype=float)
                for s_i, p in probs_arr:
                    try:
                        si = int(s_i)
                        if 0 <= si < nstates_local:
                            vec[si] = float(p)
                    except Exception:
                        continue
                probs_vec = vec
            else:
                probs_vec = np.ones(nstates_local, dtype=float) / float(max(1, nstates_local))
        else:
            probs_vec = probs_arr.astype(float)
        a_tuple = tuple(int(x) for x in a_key) if not isinstance(a_key, int) else (int(a_key),)
        transitions_compiled[(int(s_key), a_tuple)] = probs_vec
    spec.transitions_compiled = transitions_compiled

    # observations_compiled : (a_tuple, s', z_tuple) -> prob
    observations_compiled = {}
    for (s_or_sp, a_key), arr in spec.observation.items():
        try:
            arr = np.asarray(arr, dtype=float)
        except Exception:
            continue
        a_tuple = tuple(int(x) for x in a_key) if not isinstance(a_key, int) else (int(a_key),)
        for idx, p in np.ndenumerate(arr):
            if p:
                observations_compiled[(a_tuple, int(s_or_sp), tuple(int(x) for x in idx))] = float(p)
    spec.observations_compiled = observations_compiled

    # rewards_compiled : accepte (s,a)->r et (s,a,s')->r
    rewards_compiled = {}
    for key, val in spec.reward.items():
        try:
            if isinstance(key, tuple) and len(key) == 3:
                s_k, a_k, sp_k = key
                a_t = tuple(int(x) for x in a_k) if not isinstance(a_k, int) else (int(a_k),)
                rewards_compiled[(int(s_k), a_t, int(sp_k))] = float(val)
            elif isinstance(key, tuple) and len(key) == 2:
                s_k, a_k = key
                a_t = tuple(int(x) for x in a_k) if not isinstance(a_k, int) else (int(a_k),)
                rewards_compiled[(int(s_k), a_t)] = float(val)
        except Exception:
            continue
    spec.rewards_compiled = rewards_compiled

    # helpers utiles
    spec.n_actions_per_agent = [len(a) for a in spec.actions]
    spec.n_obs_per_agent = [len(o) for o in spec.observations]

    # retourner le spec construit (évite un retour implicite None)
    return spec


def parse_dpomdp_lines(lines: List[str], path: str) -> DecPOMDPSpec:
    """
    Construire un DecPOMDPSpec à partir d'une liste de lignes (extraites par ANTLR).
    Reprend la logique du parser legacy (sans lecture/écriture disque).
    """
    # copie concise de la logique legacy adaptée pour `lines`
    n_agents = 1
    discount = 1.0
    values = "reward"
    states: List[str] = []
    actions_raw = []
    observations_raw = []
    horizon: Optional[int] = None
    start_belief: Optional[np.ndarray] = None

    # interim storages
    T = {}
    O = {}
    R = {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        low = line.lower()
        if not line:
            i += 1
            continue
        if low.startswith("agents:"):
            n_agents = int(line.split(":", 1)[1].strip())
            i += 1; continue
        if low.startswith("discount:"):
            discount = float(line.split(":", 1)[1].strip())
            i += 1; continue
        if low.startswith("values:"):
            values = line.split(":", 1)[1].strip()
            i += 1; continue
        if low.startswith("states:"):
            states = line.split(":", 1)[1].strip().split()
            i += 1; continue
        if low.startswith("actions:"):
            acts = line.split(":", 1)[1].strip()
            if acts == "":
                collected = []
                j = i + 1
                while j < len(lines) and ":" not in lines[j]:
                    collected.append(lines[j].strip())
                    j += 1
                actions_raw = [c.split() for c in collected] if collected else []
                i = j; continue
            else:
                actions_raw = acts.split()
                i += 1; continue
        if low.startswith("observations:"):
            obs = line.split(":", 1)[1].strip()
            if obs == "":
                collected = []
                j = i + 1
                while j < len(lines) and ":" not in lines[j]:
                    collected.append(lines[j].strip())
                    j += 1
                observations_raw = [c.split() for c in collected] if collected else []
                i = j; continue
            else:
                observations_raw = obs.split()
                i += 1; continue
        if low.startswith("start:"):
            rest = line.split(":", 1)[1].strip()
            if rest == "" and i + 1 < len(lines) and ":" not in lines[i + 1]:
                rest = lines[i + 1].strip(); i += 1
            parts = rest.split() if rest else []
            try:
                if parts:
                    start_belief = np.array([float(x) for x in parts], dtype=np.float32)
            except Exception:
                start_belief = None
            i += 1; continue
        if low.startswith("horizon:"):
            try:
                horizon = int(line.split(":", 1)[1].strip())
            except Exception:
                horizon = None
            i += 1; continue
        i += 1

    # validations minimalistes
    if not actions_raw:
        raise ValueError(f"No 'actions' section parsed in dpomdp content: {path}")
    if not observations_raw:
        raise ValueError(f"No 'observations' section parsed in dpomdp content: {path}")

    # normalize per-agent lists
    if isinstance(actions_raw[0], list):
        actions = actions_raw
    else:
        actions = [actions_raw for _ in range(n_agents)]
    if isinstance(observations_raw[0], list):
        observations = observations_raw
    else:
        observations = [observations_raw for _ in range(n_agents)]

    # expand numeric states/actions/observations
    if len(states) == 1 and states[0].isdigit():
        n_st = int(states[0]); states = [str(i) for i in range(n_st)]
    for ag in range(len(actions)):
        al = actions[ag]
        if len(al) == 1 and al[0].isdigit():
            m = int(al[0]); actions[ag] = [str(i) for i in range(m)]
    for ag in range(len(observations)):
        ol = observations[ag]
        if len(ol) == 1 and ol[0].isdigit():
            m = int(ol[0]); observations[ag] = [str(i) for i in range(m)]

    n_states = len(states)
    if start_belief is None:
        start_belief = np.ones(n_states, dtype=np.float32) / float(max(1, n_states))

    # maps
    s2i = {s: i for i, s in enumerate(states)}
    a2i = [{a: i for i, a in enumerate(actions[ag])} for ag in range(n_agents)]
    z2i = [{z: i for i, z in enumerate(observations[ag])} for ag in range(n_agents)]

    def parse_joint_action(tokens):
        if len(tokens) != n_agents:
            raise ValueError("Joint-action length mismatch")
        out = []
        for ag, tok in enumerate(tokens):
            if isinstance(tok, int) or (isinstance(tok, str) and tok.isdigit()):
                out.append(int(tok))
            else:
                out.append(a2i[ag][tok])
        return tuple(out)

    # second pass: parse T/O/R lines from the provided `lines`
    all_joints = list(product(*[range(len(actions[ag])) for ag in range(n_agents)]))
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1; continue
        if line.startswith("T:"):
            parts = [p.strip() for p in line[2:].strip().split(":")]
            if len(parts) >= 4:
                prob_tokens = parts[3].split()
                # identify action/state tokens (reuse logic simple)
                # compute a_joint and assign probs (keep as numpy array)
                # simplified: assume parts format "s_from : a_tokens : s_to : probs"
                try:
                    s_from = parts[0] if parts[0] != "*" else "*"
                    a_tokens = parts[1].split()
                    probs = np.array([float(x) for x in prob_tokens], dtype=np.float32)
                    if a_tokens and a_tokens[0] == "*":
                        if s_from == "*":
                            for si in range(n_states):
                                for a_joint in all_joints:
                                    T[(si, a_joint)] = probs
                        else:
                            si = s2i[s_from]
                            for a_joint in all_joints:
                                T[(si, a_joint)] = probs
                    else:
                        a_joint = parse_joint_action(a_tokens)
                        if s_from == "*":
                            for si in range(n_states):
                                T[(si, a_joint)] = probs
                        else:
                            si = s2i[s_from]
                            T[(si, a_joint)] = probs
                except Exception:
                    pass
            idx += 1; continue

        if line.startswith("O:"):
            parts = [p.strip() for p in line[2:].strip().split(":")]
            if len(parts) >= 4:
                prob_part = parts[3].split()
                a_tokens = parts[1].split()
                o_part = parts[2].split()
                s_part = parts[0]
                try:
                    target_joints = all_joints if (a_tokens and a_tokens[0] == "*") else [parse_joint_action(a_tokens)]
                    if s_part == "*":
                        for si in range(n_states):
                            joint_shape = tuple(len(observations[ag]) for ag in range(n_agents))
                            arr = np.zeros(joint_shape, dtype=np.float32)
                            # assign first prob to one observed tuple if provided
                            if len(o_part) == n_agents:
                                idxs = tuple(int(x) if x.isdigit() else z2i[ag][o_part[ag]] for ag in range(n_agents))
                                arr[idxs] = float(prob_part[0])
                            for a_joint in target_joints:
                                O[(si, a_joint)] = arr
                    else:
                        si = int(s_part) if s_part.isdigit() else s2i[s_part]
                        joint_shape = tuple(len(observations[ag]) for ag in range(n_agents))
                        arr = np.zeros(joint_shape, dtype=np.float32)
                        if len(o_part) == n_agents:
                            idxs = tuple(int(x) if x.isdigit() else z2i[ag][o_part[ag]] for ag in range(n_agents))
                            arr[idxs] = float(prob_part[0])
                        for a_joint in target_joints:
                            O[(si, a_joint)] = arr
                except Exception:
                    pass
            idx += 1; continue

        if line.startswith("R:"):
            parts = [p.strip() for p in line[2:].strip().split(":")]
            r_part = parts[-1].split()
            first_tokens = parts[0].split()
            try:
                if len(first_tokens) >= 1 + n_agents:
                    s_from = first_tokens[0]; a_tokens = first_tokens[1:1 + n_agents]
                elif len(first_tokens) == n_agents:
                    a_tokens = first_tokens; s_from = parts[1] if len(parts) > 1 else "*"
                else:
                    a_tokens = parts[1].split() if len(parts) > 1 else ["*"]
                    s_from = first_tokens[0] if first_tokens else "*"

                target_joints = all_joints if (a_tokens and a_tokens[0] == "*") else [parse_joint_action(a_tokens)]
                if s_from == "*":
                    for si in range(n_states):
                        for a_joint in target_joints:
                            R[(si, a_joint)] = float(r_part[0])
                else:
                    si = int(s_from) if isinstance(s_from, str) and s_from.isdigit() else s2i[s_from]
                    for a_joint in target_joints:
                        R[(si, a_joint)] = float(r_part[0])
            except Exception:
                pass
            idx += 1; continue

        idx += 1

    if horizon is None:
        horizon = 10

    spec = DecPOMDPSpec(
        n_agents=n_agents,
        states=states,
        actions=actions,
        observations=observations,
        start=start_belief,
        start_belief=start_belief,
        transition=T,
        observation=O,
        reward=R,
        discount=discount,
        horizon=horizon,
        values=values,
        source_path=path,
    )

    # compatibility aliases and compiled tables
    spec.T = spec.transition
    spec.O = spec.observation
    spec.R = spec.reward
    # spec.start contient la croyance initiale normalisée fournie par le parser
    spec.init_belief = spec.start

    # build transitions_compiled as mapping (s, a_tuple) -> probability vector (np.ndarray)
    transitions_compiled = {}
    nstates_local = int(getattr(spec, "n_states", len(getattr(spec, "states", []))))
    for (s_key, a_key), probs in spec.transition.items():
        try:
            probs_arr = np.asarray(probs, dtype=float).ravel()
        except Exception:
            probs_arr = np.ones(nstates_local, dtype=float) / float(max(1, nstates_local))
        # if length mismatch, try sparse (s', p) pairs or fallback to uniform
        if probs_arr.size != nstates_local:
            if probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
                vec = np.zeros(nstates_local, dtype=float)
                for s_i, p in probs_arr:
                    try:
                        si = int(s_i)
                        if 0 <= si < nstates_local:
                            vec[si] = float(p)
                    except Exception:
                        continue
                probs_vec = vec
            else:
                probs_vec = np.ones(nstates_local, dtype=float) / float(max(1, nstates_local))
        else:
            probs_vec = probs_arr.astype(float)
        a_tuple = tuple(int(x) for x in a_key) if not isinstance(a_key, int) else (int(a_key),)
        transitions_compiled[(int(s_key), a_tuple)] = probs_vec
    spec.transitions_compiled = transitions_compiled

    observations_compiled = {}
    for (s_or_sp, a_key), arr in spec.observation.items():
        try:
            arr = np.asarray(arr, dtype=float)
        except Exception:
            continue
        a_tuple = tuple(int(x) for x in a_key) if not isinstance(a_key, int) else (int(a_key),)
        for idx2, p in np.ndenumerate(arr):
            if p:
                observations_compiled[(a_tuple, int(s_or_sp), tuple(int(x) for x in idx2))] = float(p)
    spec.observations_compiled = observations_compiled

    rewards_compiled = {}
    for key, val in spec.reward.items():
        try:
            if isinstance(key, tuple) and len(key) == 3:
                s_k, a_k, sp_k = key
                a_t = tuple(int(x) for x in a_k) if not isinstance(a_k, int) else (int(a_k),)
                rewards_compiled[(int(s_k), a_t, int(sp_k))] = float(val)
            elif isinstance(key, tuple) and len(key) == 2:
                s_k, a_k = key
                a_t = tuple(int(x) for x in a_k) if not isinstance(a_k, int) else (int(a_k),)
                rewards_compiled[(int(s_k), a_t)] = float(val)
        except Exception:
            continue
    spec.rewards_compiled = rewards_compiled

    spec.n_actions_per_agent = [len(a) for a in spec.actions]
    spec.n_obs_per_agent = [len(o) for o in spec.observations]

    # retourner le spec construit (évite un retour implicite None)
    return spec

# expose parse_dpomdp name expected by callers/tests
try:
    parse_dpomdp  # type: ignore
except NameError:
    # if a top-level parse exists, alias it; otherwise fallback to legacy
    if "parse" in globals():
        parse_dpomdp = parse  # type: ignore
    else:
        def parse_dpomdp(path: str) -> DecPOMDPSpec:
            return _legacy_parse_dpomdp(path)
