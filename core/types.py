# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Callable, Optional

# A sub-stage is u = (i,t) but we store it as an integer index in a DAG-ordered list.
# We keep the explicit (agent, time) too for clarity.

@dataclass(frozen=True)
class SubStage:
    idx: int
    agent: int
    time: int
    successor: Optional[int]  # index in the same list, None for terminal

@dataclass
class SubStageDAG:
    """Reverse-topological DAG of sub-stages u^{(1)} = terminal ... u^{(M)} = (1,0)."""
    stages: List[SubStage]

    def reverse_order(self) -> List[SubStage]:
        return self.stages  # already in reverse order

    def successor_of(self, uidx: int) -> Optional[int]:
        return self.stages[uidx].successor

def make_sequential_dag(n_agents: int, horizon: int) -> SubStageDAG:
    """
    Build the ordered list:
      u^{(1)} = (1, T) terminal
      u^{(2)} = (n, T-1)
      ...
      last = (1,0)
    Index = position in this list.
    """
    stages: List[SubStage] = []
    idx = 0
    # terminal
    stages.append(SubStage(idx=idx, agent=1, time=horizon, successor=None))
    idx += 1
    for t in reversed(range(horizon)):
        for i in reversed(range(1, n_agents + 1)):
            succ = None
            if i < n_agents:
                # successor is next agent at same t
                succ_time = t
                succ_agent = i + 1
                # find its future index: current idx + (i - something) is messy;
                # easier: we will fill after
            # we can't assign successor here easily; do it in second pass
            stages.append(SubStage(idx=idx, agent=i, time=t, successor=None))
            idx += 1
    # second pass to fill successors
    # stages[0] is terminal
    map_by_ti: Dict[Tuple[int, int], int] = {}
    for s in stages:
        map_by_ti[(s.agent, s.time)] = s.idx
    for s in stages[1:]:
        if s.time == horizon:
            # they should not exist
            continue
        if s.agent < n_agents:
            succ_idx = map_by_ti[(s.agent + 1, s.time)]
        else:
            # go to next stage time+1, agent=1, or terminal if time+1 == horizon
            if s.time + 1 == horizon:
                succ_idx = 0  # terminal
            else:
                succ_idx = map_by_ti[(1, s.time + 1)]
        stages[s.idx] = SubStage(
            idx=s.idx,
            agent=s.agent,
            time=s.time,
            successor=succ_idx,
        )
    return SubStageDAG(stages=stages)


def priv_projection(type_tuple: Dict[str, Any], agent: int, m: int) -> Tuple:
    """
    Project a full type (theta_u) into the private history h for agent.
    We assume type_tuple = {
        "obs": List[List[int or float]],  # per agent
        "acts": List[List[int]],          # per agent
        "t": int
    }
    We compress last m obs and last m-1 acts if present.
    """
    obs_hist = type_tuple["obs"][agent - 1]
    act_hist = type_tuple["acts"][agent - 1]
    t = type_tuple["t"]
    # last m obs
    last_obs = tuple(obs_hist[max(0, t - m + 1): t + 1])
    # last m-1 actions
    last_acts = tuple(act_hist[max(0, t - m + 1): t]) if m > 0 else ()
    return (last_obs, last_acts)