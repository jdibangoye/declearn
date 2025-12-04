# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import os
import pytest
from declearn.envs.masplan import parse_dpomdp

SPEC_NAMES = [
    "dectiger.dpomdp",
    "broadcastChannel.dpomdp",
    "recycling.dpomdp",
    "boxPushingUAI07.dpomdp",
    "GridSmall.dpomdp",
]

def _spec_path(name: str) -> str:
    root = os.path.dirname(os.path.dirname(__file__))  # declearn/
    return os.path.join(root, "envs", "masplan_specs", name)

def test_parse_many_specs():
    for name in SPEC_NAMES:
        path = _spec_path(name)
        if not os.path.exists(path):
            pytest.skip(f"Spec file missing, skip: {name}")
        spec = parse_dpomdp(path)
        assert spec is not None, f"parse returned None for {name}"
        # checks minimales de cohérence
        assert hasattr(spec, "n_agents")
        assert hasattr(spec, "n_states")
        summary = getattr(spec, "summary", None)
        if summary is not None:
            s = spec.summary()
            assert isinstance(s, str) and len(s) > 0
        # checks spécifiques (pour recycling on vérifie valeurs connues)
        if name == "recycling.dpomdp":
            assert spec.n_agents == 2
            assert spec.n_states == 4
            assert abs(getattr(spec, "discount", 1.0) - 0.9) < 1e-9