# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
"""Diagnostic des problèmes de convergence TPI"""

import pytest
import numpy as np
from declearn.core.tabular_tpi import TabularTPI
from declearn.core.sa_schedules import ThreeTimeScale
from declearn.tests.test_tabular_tpi import MinimalDAG
from declearn.envs.masplan import parse_dpomdp
from declearn.core.sequential_env import SequentialDecPOMDPSimulator
from pathlib import Path


def test_tpi_convergence_diagnostics():
    """Test de diagnostic pour identifier les problèmes de convergence TPI"""
    
    # Configuration avec taux d'apprentissage plus appropriés
    print("\n=== DIAGNOSTIC TPI CONVERGENCE ===")
    
    # Test 1: Taux d'apprentissage
    print("\n1. Test des taux d'apprentissage:")
    
    # Timescale minimal (problématique)
    from declearn.tests.test_tabular_tpi import MinimalTimescale
    ts_minimal = MinimalTimescale()
    rates_minimal = ts_minimal.decay(100)
    print(f"  MinimalTimescale k=100: α={rates_minimal.alpha}, β={rates_minimal.beta}, γ={rates_minimal.gamma}")
    
    # Timescale théorique (correct)
    ts_proper = ThreeTimeScale(alpha=1.0, beta=0.1, gamma=0.01)
    rates_proper = ts_proper.decay(100) 
    print(f"  ThreeTimeScale k=100:   α={rates_proper.alpha:.6f}, β={rates_proper.beta:.6f}, γ={rates_proper.gamma:.6f}")
    
    # Test 2: Nombre d'épisodes nécessaires
    print("\n2. Test sur environnement simple:")
    
    # Environnement très simple pour tester convergence
    dag = MinimalDAG(1)
    n_actions = {0: 2}
    rng = np.random.default_rng(42)
    
    # TPI avec timescale correct
    tpi = TabularTPI(dag, n_actions, ts_proper, 1, rng)
    
    # Simulation d'apprentissage sur problème simple
    simple_rewards = []
    for episode in range(50):
        # Sample très simple et répétitif pour forcer l'apprentissage
        from declearn.core.sequential_env import TypeTuple
        
        theta = TypeTuple(
            state=episode % 2, t=0,
            joint_histories=[{'obs': [episode % 2], 'acts': [-1]}],
            prefix_actions=[]
        )
        theta_next = TypeTuple(
            state=(episode + 1) % 2, t=1,
            joint_histories=[{'obs': [episode % 2, (episode + 1) % 2], 'acts': [-1, 0]}],
            prefix_actions=[]
        )
        
        sample = {
            0: {
                "theta": theta,
                "a": 0,
                "r": 1.0,  # Reward constant et positif
                "theta_next": theta_next,
                "h": (episode % 2,),
                "h_next": ((episode + 1) % 2,),
                "a_next": 0,
                "u": 0
            }
        }
        
        tpi.step(sample, k=episode)
        
        # Evaluation périodique
        if episode % 10 == 9:
            # Utiliser nouvelle structure u=(agent, t)
            all_q_values = []
            for u_key, q_table in tpi.q_tables.items():
                for q_vals in q_table.values():
                    all_q_values.extend(q_vals)
            avg_q = np.mean(all_q_values) if all_q_values else 0
            total_entries = sum(len(q_table) for q_table in tpi.q_tables.values())
            print(f"  Episode {episode+1}: Q-table avg={avg_q:.4f}, entries={total_entries}")
            simple_rewards.append(avg_q)
    
    # Test 3: Exploration vs exploitation
    print("\n3. Test exploration/exploitation:")
    
    actions_greedy = []
    actions_stochastic = []
    test_theta = TypeTuple(
        state=0, t=0,
        joint_histories=[{'obs': [0], 'acts': [-1]}],
        prefix_actions=[]
    )
    
    for _ in range(20):
        action_greedy = tpi.act(0, test_theta, greedy=True)
        action_stochastic = tpi.act(0, test_theta, greedy=False)
        actions_greedy.append(action_greedy)
        actions_stochastic.append(action_stochastic)
    
    print(f"  Actions greedy: {set(actions_greedy)} (diversité: {len(set(actions_greedy))})")
    print(f"  Actions stochastic: {set(actions_stochastic)} (diversité: {len(set(actions_stochastic))})")
    
    # Assertions pour validation
    total_q_entries = sum(len(q_table) for q_table in tpi.q_tables.values())
    assert total_q_entries > 0, "Q-tables doivent être remplies après entraînement"
    assert len(simple_rewards) > 0, "Doit avoir des récompenses enregistrées"
    
    print(f"\n✅ Diagnostic terminé. Q-tables: {total_q_entries} entrées")


def test_tpi_benchmark_improved_params():
    """Test avec paramètres améliorés sur un vrai benchmark"""
    
    print("\n=== TEST PARAMÈTRES AMÉLIORÉS ===")
    
    # Charger un environnement simple
    specs_dir = Path(__file__).parent.parent / "envs" / "masplan_specs"
    dpomdp_path = specs_dir / "GridSmall.dpomdp"
    
    if not dpomdp_path.exists():
        pytest.skip(f"Spec file not found: {dpomdp_path}")
    
    spec = parse_dpomdp(str(dpomdp_path))
    
    # Configuration améliorée
    dag = MinimalDAG(spec.n_agents)
    n_actions = {i: len(spec.actions[i]) for i in range(spec.n_agents)}
    
    # CORRECTION 1: Utiliser vrais paramètres théoriques
    ts_improved = ThreeTimeScale(alpha=0.5, beta=0.1, gamma=0.02)
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(dag, n_actions, ts_improved, 1, rng)
    sim = SequentialDecPOMDPSimulator(spec, seed=42, memory_m=1)
    
    # CORRECTION 2: Plus d'épisodes d'entraînement
    episodes = 1000  # Au lieu de 300
    training_rewards = []
    
    print(f"Entraînement sur {episodes} épisodes...")
    
    for episode in range(episodes):
        type_tuple = sim.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < 20:  # Horizon plus étendu
            # Simulateur séquentiel : un agent agit à la fois
            agent_id = type_tuple.current_agent if hasattr(type_tuple, 'current_agent') else step % spec.n_agents
            
            # Plus exploratoire au début (80% exploration, puis 20%)  
            use_greedy = episode > episodes * 0.8
            action = tpi.act(agent_id, type_tuple, greedy=use_greedy)
            
            # Step séquentiel
            next_type_tuple, reward, done, _ = sim.step(action)
            episode_reward += reward
            
            # Mise à jour TPI avec sample simple
            if step > 0:  # Éviter premier step
                theta = (type_tuple.state, type_tuple.t) 
                theta_next = (next_type_tuple.state, next_type_tuple.t) if not done else theta
                h = tuple(type_tuple.joint_histories[agent_id]['obs'][-2:]) if type_tuple.joint_histories else (type_tuple.state,)
                h_next = tuple(next_type_tuple.joint_histories[agent_id]['obs'][-2:]) if not done and next_type_tuple.joint_histories else h
                
                sample = {
                    agent_id: {
                        "theta": theta,
                        "a": action,
                        "r": float(reward),
                        "theta_next": theta_next,
                        "h": h,
                        "h_next": h_next,
                        "a_next": 0,  # Placeholder
                        "u": agent_id
                    }
                }
                
                tpi.step(sample, k=episode * 20 + step)
            
            type_tuple = next_type_tuple
            step += 1
        
        training_rewards.append(episode_reward)
        
        if episode % 200 == 199:
            recent_avg = np.mean(training_rewards[-50:])
            print(f"  Episode {episode+1}: reward avg (50 derniers) = {recent_avg:.3f}")
    
    # Évaluation finale
    print("\nÉvaluation finale...")
    eval_rewards = []
    for _ in range(50):
        type_tuple = sim.reset()
        done = False
        episode_reward = 0
        step = 0
        
        while not done and step < 20:
            agent_id = type_tuple.current_agent if hasattr(type_tuple, 'current_agent') else step % spec.n_agents
            action = tpi.act(agent_id, type_tuple, greedy=True)
            
            next_type_tuple, reward, done, _ = sim.step(action)
            episode_reward += reward
            type_tuple = next_type_tuple
            step += 1
        
        eval_rewards.append(episode_reward)
    
    final_performance = np.mean(eval_rewards)
    print(f"Performance finale: {final_performance:.4f}")
    
    # Statistiques d'apprentissage
    print(f"Q-tables créées: {sum(len(qt) for qt in tpi.q_tables.values())}")
    print(f"G-tables créées: {len(tpi.g_tables)}")
    
    # Validation d'amélioration
    assert final_performance > -10, "Performance doit être raisonnable"
    total_q_states = sum(len(qt) for qt in tpi.q_tables.values())
    assert total_q_states > 5, "Doit avoir exploré plusieurs états"


if __name__ == "__main__":
    test_tpi_convergence_diagnostics()
    test_tpi_benchmark_improved_params()