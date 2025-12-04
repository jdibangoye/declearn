# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
class DummyMAPPO:
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def fit(self, batch):
        pass

    def evaluate(self, env, episodes: int = 10):
        return 0.0