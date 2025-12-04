# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import os
from declearn.envs.masplan import parse_dpomdp

def test_parse_broadcast_channel():
    pkg_root = os.path.dirname(os.path.dirname(__file__))  # declearn/
    spec_path = os.path.join(pkg_root, "envs", "masplan_specs", "broadcastChannel.dpomdp")
    assert os.path.exists(spec_path), f"Missing spec file: {spec_path}"
    spec = parse_dpomdp(spec_path)
    assert spec is not None
    assert spec.n_agents == 2
    assert spec.n_states == 4
    assert [len(a) for a in spec.actions] == [2, 2]
    assert [len(z) for z in spec.observations] == [2, 2]
    assert spec.discount == 1.0
    assert isinstance(spec.summary(), str) and len(spec.summary()) > 0