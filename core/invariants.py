# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from typing import Dict, Tuple, Any

def check_td_well_formed(sample: Dict[str, Any]) -> None:
    assert "theta" in sample and "a" in sample and "r" in sample and "theta_next" in sample, \
        "TD sample missing keys"

def check_locality(u_idx: int, sample_u_idx: int) -> None:
    assert u_idx == sample_u_idx, "cross-substage contamination detected"