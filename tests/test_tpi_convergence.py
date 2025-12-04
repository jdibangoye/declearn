# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import numpy as np
import matplotlib.pyplot as plt
from declearn.core.tabular_tpi import TabularTPI
from declearn.tests.test_tabular_tpi import MinimalDAG
from declearn.core.sa_schedules import ThreeTimeScale
from declearn.core.sequential_env import TypeTuple

def test_tpi_convergence():
    """Test de convergence TPI sur plusieurs épisodes"""
    
    dag = MinimalDAG(1)
    n_actions = {0: 2}
    ts = ThreeTimeScale(alpha=0.5, beta=0.1, gamma=0.01)
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(dag, n_actions, ts, memory_m=2, rng=rng)
    
    print("=== TEST CONVERGENCE TPI ===")
    
    # Métriques de convergence
    q_variances = []
    g_sizes = []
    policy_entropies = []
    rewards = []
    
    # Simulation sur 50 épisodes
    for episode in range(50):
        
        # Générer échantillon aléatoire mais cohérent
        state = episode % 3
        reward = float(episode % 2)  # Alternance rewards
        
        sample = {
            0: {
                "theta": TypeTuple(
                    state=state, t=episode,
                    joint_histories=[{'obs': [state, (state+1)%3], 'acts': [-1, episode%2]}],
                    prefix_actions=[]
                ),
                "a": episode % 2, "r": reward,
                "theta_next": TypeTuple(
                    state=(state+1)%3, t=episode+1,
                    joint_histories=[{'obs': [state, (state+1)%3, (state+2)%3], 'acts': [-1, episode%2, (episode+1)%2]}],
                    prefix_actions=[]
                ),
                "h": (state, (state+1)%3), "u": 0
            }
        }
        
        tpi.step(sample, k=episode)
        
        # Collecter métriques tous les 5 épisodes
        if episode % 5 == 0:
            
            # Variance des Q-values - utiliser nouvelle structure
            all_q_values = []
            for u_key, q_table in tpi.q_tables.items():
                for q_vals in q_table.values():
                    all_q_values.extend(q_vals)
            q_var = np.var(all_q_values) if all_q_values else 0
            q_variances.append(q_var)
            
            # Taille des G-tables
            g_stats = tpi.get_g_table_stats()
            total_g_entries = sum(stats['total_entries'] for stats in g_stats.values())
            g_sizes.append(total_g_entries)
            
            # Entropie de la politique
            mock_theta = TypeTuple(state=1, t=episode, joint_histories=[{'obs': [1,2]}], prefix_actions=[])
            # Approximation entropie via échantillonnage
            actions_sample = [tpi.act(0, mock_theta, greedy=False) for _ in range(100)]
            action_counts = np.bincount(actions_sample, minlength=n_actions[0])
            action_probs = action_counts / 100
            entropy = -sum(p * np.log(p + 1e-10) for p in action_probs if p > 0)
            policy_entropies.append(entropy)
            
            rewards.append(reward)
            
            # Log progrès
            rates = tpi.get_learning_rate_diagnostics(episode)
            print(f"Épisode {episode:2d}: Q-var={q_var:.4f}, G-size={total_g_entries:2d}, "
                  f"Entropy={entropy:.3f}, α={rates['alpha_k']:.4f}")
    
    print(f"\n=== ANALYSE CONVERGENCE ===")
    print(f"Q-variance finale: {q_variances[-1]:.6f} (initial: {q_variances[0]:.6f})")
    print(f"G-tables finale: {g_sizes[-1]} entrées (initial: {g_sizes[0]})")
    print(f"Entropie finale: {policy_entropies[-1]:.3f} (initial: {policy_entropies[0]:.3f})")
    
    # Détection convergence
    if len(q_variances) >= 3:
        recent_q_change = abs(q_variances[-1] - q_variances[-3])
        converged = recent_q_change < 0.001
        print(f"Convergence Q-values: {'✅' if converged else '❌'} (changement: {recent_q_change:.6f})")

if __name__ == "__main__":
    test_tpi_convergence()