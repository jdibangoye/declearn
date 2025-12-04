# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from declearn.core.sa_schedules import ThreeTimeScale

def test_decay_separation():
    ts = ThreeTimeScale(0.5, 0.1, 0.02)
    ts2 = ts.decay(100)
    assert ts2.alpha > ts2.beta
    assert ts2.beta > ts2.gamma

