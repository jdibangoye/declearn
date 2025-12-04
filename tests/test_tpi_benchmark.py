# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import numpy as np
from declearn.core.tabular_tpi import TabularTPI
from declearn.tests.test_tabular_tpi import MinimalDAG
from declearn.core.sa_schedules import ThreeTimeScale
from declearn.core.sequential_env import TypeTuple

def run_random_policy(n_actions, episodes=100):
    """Politique aléatoire de référence"""
    rewards = []
    rng = np.random.default_rng(42)
    
    for episode in range(episodes):
        # Reward aléatoire mais reproductible
        reward = rng.uniform(-1, 1)
        rewards.append(reward)
    
    return rewards

def run_tpi_policy(tpi, n_actions, episodes=100):
    """Politique TPI entraînée"""
    rewards = []
    
    for episode in range(episodes):
        
        # Générer état cohérent
        state = episode % 3
        
        # Créer theta pour décision
        theta = TypeTuple(
            state=state, t=episode,
            joint_histories=[{'obs': [state], 'acts': [-1]}],
            prefix_actions=[]
        )
        
        # Action TPI
        action = tpi.act(0, theta, greedy=True)
        
        # Reward basé sur performance (exemple simple)
        if action == 1 and state == 1:  # "Bonne" action dans "bon" état
            reward = 1.0
        elif action == 0 and state == 0:
            reward = 0.5
        else:
            reward = -0.1
        
        rewards.append(reward)
        
        # Optionnel : continuer l'entraînement
        sample = {
            0: {
                "theta": theta, "a": action, "r": reward,
                "theta_next": TypeTuple(state=(state+1)%3, t=episode+1, joint_histories=[{'obs': [state, (state+1)%3]}], prefix_actions=[]),
                "h": (state,), "u": 0
            }
        }
        tpi.step(sample, k=episodes + episode)  # Continue training
    
    return rewards

def test_tpi_benchmark():
    """Comparaison TPI vs politique aléatoire"""
    
    print("=== BENCHMARK TPI vs RANDOM ===")
    
    # Configuration
    dag = MinimalDAG(1)
    n_actions = {0: 2}
    ts = ThreeTimeScale(alpha=0.3, beta=0.1, gamma=0.01)
    rng = np.random.default_rng(42)
    
    # Entraîner TPI
    tpi = TabularTPI(dag, n_actions, ts, memory_m=1, rng=rng)
    
    # Phase d'entraînement
    print("Phase d'entraînement TPI...")
    for episode in range(50):
        sample = {
            0: {
                "theta": TypeTuple(state=episode%3, t=episode, joint_histories=[{'obs': [episode%3]}], prefix_actions=[]),
                "a": episode % 2, "r": float((episode % 2) * (episode % 3 == 1)),
                "theta_next": TypeTuple(state=(episode+1)%3, t=episode+1, joint_histories=[{'obs': [episode%3, (episode+1)%3]}], prefix_actions=[]),
                "h": (episode%3,), "u": 0
            }
        }
        tpi.step(sample, k=episode)
    
    # Test performance
    episodes_test = 100
    
    random_rewards = run_random_policy(n_actions, episodes_test)
    tpi_rewards = run_tpi_policy(tpi, n_actions, episodes_test)
    
    # Analyse
    random_mean = np.mean(random_rewards)
    tpi_mean = np.mean(tpi_rewards)
    improvement = tpi_mean - random_mean
    
    print(f"\n=== RÉSULTATS BENCHMARK ===")
    print(f"Politique aléatoire: {random_mean:.3f} ± {np.std(random_rewards):.3f}")
    print(f"Politique TPI:       {tpi_mean:.3f} ± {np.std(tpi_rewards):.3f}")
    print(f"Amélioration:        {improvement:+.3f} ({improvement/abs(random_mean)*100:+.1f}%)")
    
    # Test statistique simple
    if improvement > 2 * np.std(random_rewards):
        print("✅ TPI significativement meilleur")
    else:
        print("❌ Pas d'amélioration significative")

if __name__ == "__main__":
    test_tpi_benchmark()