# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from dataclasses import dataclass

@dataclass
class ThreeTimeScale:
    alpha: float  # fast
    beta: float   # slower
    gamma: float  # slowest

    def decay(self, k: int) -> "ThreeTimeScale":
        # simple polyak-style decay, but keeping separation
        return ThreeTimeScale(
            alpha=self.alpha / (1.0 + 1e-4 * k),
            beta=self.beta / (1.0 + 5e-4 * k),
            gamma=self.gamma / (1.0 + 1e-3 * k),
        )

@dataclass  
class ExplorationSchedule:
    """Schedule d'exploration ε-greedy pour TPI avec comportement GLIE"""
    initial_epsilon: float = 0.3  # Plus d'exploration initiale
    decay_rate: float = 5e-4      # Décroissance plus lente
    
    def epsilon(self, k: int) -> float:
        """Calcule ε_k qui décroît vers 0 pour GLIE (Greedy in the Limit with Infinite Exploration)"""
        return self.initial_epsilon / (1.0 + self.decay_rate * k)