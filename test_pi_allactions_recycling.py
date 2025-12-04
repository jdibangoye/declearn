# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.

"""
Test de Policy Iteration AllActions sur Recycling T=1

Devrait converger vers V=5.0 (politique optimale waitandrecharge pour les 2 agents)
"""

import numpy as np
from declearn.envs.masplan import parse_dpomdp
from declearn.core.sequential_env import SequentialDecPOMDPSimulator
from declearn.core.policy_iteration_allactions import PolicyIterationAllActions
from declearn.core.types import make_sequential_dag

# ============================================================================
# 1. Charger Recycling
# ============================================================================

spec = parse_dpomdp("declearn/envs/masplan_specs/recycling.dpomdp")
spec.horizon = 1  # T=1

print(f"Spec chargée : Recycling")
print(f"  Agents : {spec.n_agents}")
print(f"  Actions : {spec.actions}")
print(f"  États : {len(spec.states)} états")
print(f"  Horizon : {spec.horizon}")

# ============================================================================
# 2. Créer simulateur factory
# ============================================================================

def sim_factory():
    return SequentialDecPOMDPSimulator(
        spec=spec,
        memory_m=0,  # Pas besoin de mémoire à T=1
        seed=None
    )

# Tester qu'un simulateur fonctionne
test_sim = sim_factory()
test_theta = test_sim.reset()
print(f"\n✓ Simulateur créé, état initial : {test_theta}")

# ============================================================================
# 3. Calculer valeur optimale exacte (exhaustive search)
# ============================================================================

def exhaustive_search_horizon_1(spec):
    """Recherche exhaustive de la politique optimale à T=1"""
    
    n_agents = spec.n_agents
    n_actions_per_agent = [len(spec.actions[i]) for i in range(n_agents)]
    
    best_value = -float('inf')
    best_policy = None
    
    # Énumérer toutes les politiques jointes possibles
    def enumerate_policies(agent_idx):
        if agent_idx == n_agents:
            # Politique jointe complète, calculer sa valeur
            nonlocal best_value, best_policy
            
            # Simuler avec cette politique
            sim = sim_factory()
            theta = sim.reset()
            
            total_reward = 0.0
            for i in range(n_agents):
                action = current_policy[i]
                theta, reward, done, _ = sim.step(action)
                total_reward += reward
            
            if total_reward > best_value:
                best_value = total_reward
                best_policy = current_policy.copy()
            
            return
        
        # Essayer chaque action pour cet agent
        for a in range(n_actions_per_agent[agent_idx]):
            current_policy[agent_idx] = a
            enumerate_policies(agent_idx + 1)
    
    current_policy = [0] * n_agents
    enumerate_policies(0)
    
    return best_value, best_policy

print("\n" + "="*60)
print("RECHERCHE EXHAUSTIVE (baseline)")
print("="*60)

v_star, pi_star = exhaustive_search_horizon_1(spec)
print(f"Valeur optimale V* = {v_star:.4f}")
print(f"Politique optimale π* = {pi_star}")

# Afficher noms des actions
action_names = []
for i, a in enumerate(pi_star):
    if isinstance(spec.actions[i], list):
        name = spec.actions[i][a]
    else:
        name = f"action_{a}"
    action_names.append(name)
print(f"  → {' + '.join(action_names)}")

# ============================================================================
# 4. Créer n_actions_dict
# ============================================================================

n_actions_dict = {}
for i in range(spec.n_agents):
    if isinstance(spec.actions[i], list):
        n_actions_dict[i] = len(spec.actions[i])
    else:
        n_actions_dict[i] = spec.actions[i]

print(f"\nn_actions_dict = {n_actions_dict}")

# ============================================================================
# 5. Entraîner avec Policy Iteration AllActions
# ============================================================================

print("\n" + "="*60)
print("POLICY ITERATION ALLACTIONS")
print("="*60)

# Créer le DAG
dag = make_sequential_dag(n_agents=spec.n_agents, horizon=spec.horizon)

print(f"DAG : {len(dag.stages)} substages")
for i, stage in enumerate(dag.stages):
    print(f"  Stage {i}: agent={stage.agent}, time={stage.time}, successor={stage.successor}")

# Créer l'algorithme
pi_algo = PolicyIterationAllActions(
    dag=dag,
    n_actions_dict=n_actions_dict,
    n_iterations=20,        # 20 itérations maximum
    n_eval_episodes=50000,  # 50000 épisodes d'évaluation par itération (10x plus)
    alpha=0.1,              # Learning rate Q
    beta=0.05,              # Learning rate G (augmenté pour convergence plus rapide)
    memory_m=0,             # Pas de mémoire à T=1
    rng=np.random.default_rng(42)
)

# Entraîner
print("\nEntraînement...")
policies = pi_algo.train(sim_factory, verbose=True)

# ============================================================================
# 6. Évaluer la politique apprise
# ============================================================================

print("\n" + "="*60)
print("ÉVALUATION POLITIQUE APPRISE")
print("="*60)

# Afficher G-tables
print("\nG-TABLES apprises :")
for u in sorted(pi_algo.g_tables.keys()):
    print(f"\n  Substage u={u} (agent={u[0]}, t={u[1]}):")
    for h, g_vals in pi_algo.g_tables[u].items():
        print(f"    h={h}:")
        for a in range(len(g_vals)):
            action_name = spec.actions[u[0]][a] if isinstance(spec.actions[u[0]], list) else f"a{a}"
            print(f"      {action_name}: G={g_vals[a]:.4f}")

# Simuler avec politique apprise
def evaluate_learned_policy(n_episodes=1000):
    """Évaluer la politique apprise sur n_episodes"""
    
    total_reward = 0.0
    policy_counts = {}
    
    for _ in range(n_episodes):
        sim = sim_factory()
        theta = sim.reset()
        episode_reward = 0.0
        episode_actions = []
        
        for stage in dag.stages:
            agent_id = stage.agent - 1  # 0-indexed
            action = pi_algo.act(agent_id, theta, greedy=True)
            episode_actions.append(action)
            theta, reward, done, _ = sim.step(action)
            episode_reward += reward
        
        total_reward += episode_reward
        
        # Compter les politiques jointes
        policy_tuple = tuple(episode_actions)
        policy_counts[policy_tuple] = policy_counts.get(policy_tuple, 0) + 1
    
    avg_reward = total_reward / n_episodes
    
    return avg_reward, policy_counts

avg_value, policy_counts = evaluate_learned_policy(n_episodes=1000)

print(f"\nValeur moyenne sur 1000 épisodes : V = {avg_value:.4f}")
print(f"Comparaison avec V* = {v_star:.4f}")
print(f"Erreur = {abs(avg_value - v_star) / v_star * 100:.2f}%")

print("\nPolitiques jointes observées :")
for policy, count in sorted(policy_counts.items(), key=lambda x: -x[1])[:5]:
    freq = count / 1000 * 100
    action_names_list = []
    for i, a in enumerate(policy):
        if i < len(spec.actions) and isinstance(spec.actions[i], list) and a < len(spec.actions[i]):
            action_names_list.append(spec.actions[i][a])
        else:
            action_names_list.append(f"a{a}")
    print(f"  {policy} ({' + '.join(action_names_list)}): {count}/1000 ({freq:.1f}%)")

# ============================================================================
# 7. Résumé
# ============================================================================

print("\n" + "="*60)
print("RÉSUMÉ")
print("="*60)

print(f"Valeur optimale théorique : V* = {v_star:.4f}")
print(f"Politique optimale : π* = {pi_star} ({' + '.join(action_names)})")
print(f"\nValeur apprise (PI-AllActions) : V = {avg_value:.4f}")
print(f"Erreur : {abs(avg_value - v_star) / v_star * 100:.2f}%")

if abs(avg_value - v_star) / v_star < 0.01:
    print("\n✅ SUCCÈS : Convergence vers optimum (< 1% erreur)")
else:
    print(f"\n⚠️  ÉCHEC : Pas de convergence vers optimum ({abs(avg_value - v_star) / v_star * 100:.1f}% erreur)")
