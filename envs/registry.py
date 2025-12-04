# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from typing import Any, Dict
from .masplan import parse_dpomdp
from .pettingzoo import PZWrapper, PZ_AVAILABLE
from .rware import RWAREWrapper
from .smacv2 import SMACv2Wrapper

def build_env(name: str, **kwargs) -> Any:
    if name.endswith(".dpomdp"):
        spec = parse_dpomdp(name)
        return spec
    if name in ("simple_spread_v3", "simple_tag_v3", "simple_adversary_v3"):
        assert PZ_AVAILABLE, "PettingZoo not available"
        return PZWrapper(name, horizon=kwargs.get("horizon", 25), seed=kwargs.get("seed", 0))
    if name.startswith("rware"):
        return RWAREWrapper(layout=name, horizon=kwargs.get("horizon", 50), seed=kwargs.get("seed", 0))
    if name.startswith("smac"):
        return SMACv2Wrapper(map_name=name, horizon=kwargs.get("horizon", 60), seed=kwargs.get("seed", 0))
    raise ValueError(f"Unknown env {name}")