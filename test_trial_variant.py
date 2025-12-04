# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
"""
Test de la variante Trial (propagation arrière récursive)
"""

from declearn.envs.masplan import parse_dpomdp
from declearn.core.sequential_env import SequentialDecPOMDPSimulator
from declearn.core.types import make_sequential_dag
from declearn.core.tabular_tpi import TabularTPI
from declearn.core.sa_schedules import ThreeTimeScale
import numpy as np

path = 'declearn/envs/masplan_specs/GridSmall.dpomdp'
spec = parse_dpomdp(path)
spec.horizon = 2

dag = make_sequential_dag(spec.n_agents, 2)
n_actions = {i: len(spec.actions[i]) for i in range(spec.n_agents)}

# Utiliser des taux constants pour cette variante
ts = ThreeTimeScale(alpha=0.1, beta=0.05, gamma=0.01)
ts.alpha_base = 0.1
ts.beta_base = 0.05
ts.gamma_base = 0.01

rng = np.random.default_rng(42)
tpi = TabularTPI(dag, n_actions, ts, 1, rng)

print("=" * 80)
print("TEST : Variante Trial avec propagation arrière récursive")
print("=" * 80)

episodes = 100
rewards = []

for episode in range(episodes):
    sim = SequentialDecPOMDPSimulator(spec, seed=42 + episode, memory_m=1)
    theta_init = sim.reset()
    
    # Ligne 1-3 de l'algorithme : u ← u^(M), sample θ, Trial(u,θ)
    # u^(M) est le DERNIER nœud dans le DAG (premier dans l'ordre d'exécution)
    # Dans l'ordre topologique inverse, c'est len(dag.stages)-1
    u_start_idx = len(dag.stages) - 1  # Dernier nœud = premier agent à t=0
    
    # ε_k pour GLIE : décroit de 1.0 vers 0
    epsilon_k = max(0.01, 1.0 - episode / (episodes * 0.7))
    
    total_reward = tpi.trial(sim, u_start_idx, theta_init, k=episode, epsilon_k=epsilon_k)
    rewards.append(total_reward)
    
    if episode % 20 == 0:
        avg_reward = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
        q_entries = sum(len(q_dict) for q_dict in tpi.q_tables.values())
        g_entries = sum(len(g_dict) for g_dict in tpi.g_tables.values())
        print(f"Episode {episode:3d}: ε={epsilon_k:.3f}, R_avg={avg_reward:.3f}, "
              f"Q={len(tpi.q_tables)} tables ({q_entries} états), "
              f"G={len(tpi.g_tables)} tables ({g_entries} hist)")

print("\n" + "=" * 80)
print("RÉSULTATS FINAUX")
print("=" * 80)

final_reward = np.mean(rewards[-20:])
print(f"Récompense moyenne (derniers 20 épisodes) : {final_reward:.4f}")

# Analyse des tables Q
print("\nAnalyse des Q-tables par sous-paire:")
for u_pair in sorted(tpi.q_tables.keys()):
    q_dict = tpi.q_tables[u_pair]
    n_states = len(q_dict)
    
    all_q_values = [v for q_theta in q_dict.values() for v in q_theta if abs(v) > 1e-6]
    n_nonzero = len(all_q_values)
    
    if n_nonzero > 0:
        print(f"  u={u_pair}: {n_states} états, {n_nonzero} Q non-zéro, "
              f"Q_max={max(all_q_values):.4f}, Q_avg={np.mean(all_q_values):.4f}")
    else:
        print(f"  u={u_pair}: {n_states} états, AUCUNE propagation")

print("\n" + "=" * 80)
print("COMPARAISON")
print("=" * 80)
print("""
Benchmark GridSmall (TPI original) : 0.910
TPI avec SARSA (correction précédente) : à mesurer
TPI avec Trial (cette variante) : {:.4f}

La variante Trial devrait converger plus vite grâce à la propagation
arrière immédiate des récompenses lors de la récursion.
""".format(final_reward))
