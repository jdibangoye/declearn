# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
# Optional: only stub; real SMACv2 needs extra deps
class SMACv2Wrapper:
    def __init__(self, map_name: str = "3m", horizon: int = 60, seed: int = 0):
        self.map_name = map_name
        self.horizon = horizon
        self.seed = seed
        self.n_agents = 3
        self.n_actions = {i: 6 for i in range(self.n_agents)}

    def reset(self):
        return {}

    def step(self, actions):
        return {}, {i: 0.0 for i in range(self.n_agents)}, {i: False for i in range(self.n_agents)}, {}