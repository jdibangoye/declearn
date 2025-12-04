# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
"""Test pour vérifier le nombre réel d'états explorés par TPI"""

import pytest
import numpy as np
from declearn.envs.masplan import parse_dpomdp
from declearn.core.sequential_env import SequentialDecPOMDPSimulator
from declearn.core.tabular_tpi import TabularTPI
from declearn.tests.test_tabular_tpi import MinimalDAG, MinimalTimescale
from pathlib import Path


def test_tpi_exploration_analysis():
    """Analyser précisément l'exploration TPI sur GridSmall"""
    
    print("\n=== ANALYSE DÉTAILLÉE DE L'EXPLORATION TPI ===")
    
    # Charger GridSmall
    specs_dir = Path(__file__).parent.parent / "envs" / "masplan_specs"
    dpomdp_path = specs_dir / "GridSmall.dpomdp"
    
    if not dpomdp_path.exists():
        pytest.skip(f"Spec file not found: {dpomdp_path}")
    
    spec = parse_dpomdp(str(dpomdp_path))
    
    print(f"Spec GridSmall:")
    print(f"  - Agents: {spec.n_agents}")
    print(f"  - États: {len(spec.states)}")
    print(f"  - Actions par agent: {[len(spec.actions[i]) for i in range(spec.n_agents)]}")
    print(f"  - Observations par agent: {[len(spec.observations[i]) for i in range(spec.n_agents)]}")
    
    # Configuration TPI
    dag = MinimalDAG(spec.n_agents)
    n_actions = {i: len(spec.actions[i]) for i in range(spec.n_agents)}
    ts = MinimalTimescale()
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(dag, n_actions, ts, 1, rng)
    sim = SequentialDecPOMDPSimulator(spec, seed=42, memory_m=1)
    
    print(f"\nConfiguration TPI:")
    print(f"  - n_actions_dict: {n_actions}")
    print(f"  - Agents dans TPI: {list(tpi.q_tables.keys())}")
    
    # Simulation d'entraînement avec statistiques détaillées
    episodes = 100
    
    for episode in range(episodes):
        type_tuple = sim.reset()
        done = False
        step = 0
        
        while not done and step < 15:
            # Simulateur séquentiel
            agent_id = step % spec.n_agents
            
            # Action aléatoire pour forcer l'exploration
            if episode < 50:
                action = rng.integers(0, n_actions[agent_id])  # Exploration pure
            else:
                action = tpi.act(agent_id, type_tuple, greedy=False)  # TPI avec exploration
            
            next_type_tuple, reward, done, _ = sim.step(action)
            
            # Mise à jour TPI
            if step > 0:
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
                        "a_next": 0,
                        "u": agent_id
                    }
                }
                
                tpi.step(sample, k=episode * 15 + step)
            
            type_tuple = next_type_tuple
            step += 1
        
        # Statistiques périodiques
        if episode % 25 == 24:
            total_q_states = sum(len(q_table) for q_table in tpi.q_tables.values())
            total_g_states = sum(len(g_table) for g_table in tpi.g_tables.values())
            
            print(f"\nÉpisode {episode+1}:")
            print(f"  - Q-tables par agent: {[(agent, len(q_table)) for agent, q_table in tpi.q_tables.items()]}")
            print(f"  - G-tables par sous-paire: {[(u, len(g_table)) for u, g_table in tpi.g_tables.items()]}")
            print(f"  - Total états Q: {total_q_states}")
            print(f"  - Total états G: {total_g_states}")
    
    # Analyse finale détaillée
    print(f"\n=== ANALYSE FINALE ===")
    
    # Q-tables analysis
    print(f"Structure Q-tables:")
    for agent_id, q_table in tpi.q_tables.items():
        print(f"  Agent {agent_id}: {len(q_table)} états theta explorés")
        if len(q_table) > 0:
            print(f"    États: {list(q_table.keys())[:5]}...")  # Premiers 5 états
            print(f"    Exemple Q-values: {q_table[list(q_table.keys())[0]]}")
    
    # G-tables analysis  
    print(f"Structure G-tables:")
    for u_pair, g_table in tpi.g_tables.items():
        print(f"  Sous-paire {u_pair}: {len(g_table)} historiques h explorés")
        if len(g_table) > 0:
            print(f"    Historiques: {list(g_table.keys())[:3]}...")
    
    # Validation
    total_q_entries = sum(len(q_table) for q_table in tpi.q_tables.values())
    total_g_entries = sum(len(g_table) for g_table in tpi.g_tables.values())
    
    print(f"\nRÉSUMÉ:")
    print(f"  - Nombre d'agents TPI: {len(tpi.q_tables)}")
    print(f"  - Nombre de sous-paires G: {len(tpi.g_tables)}")
    print(f"  - États θ explorés (total): {total_q_entries}")
    print(f"  - Historiques h explorés (total): {total_g_entries}")
    
    # Vérifications adaptées à la nouvelle structure
    # Avec nouvelle structure, on a des Q-tables par sous-paire (agent, t)
    agents_in_q_tables = set(u_key[0] for u_key in tpi.q_tables.keys())
    assert len(agents_in_q_tables) <= spec.n_agents, f"Agents dans Q-tables {agents_in_q_tables} dépasse {spec.n_agents}"
    assert len(agents_in_q_tables) > 0, "Devrait avoir au moins un agent dans les Q-tables"
    assert total_q_entries > 5, f"Devrait explorer plus de 5 états theta, trouvé {total_q_entries}"
    
    print(f"\n✅ Analyse terminée. Exploration: θ={total_q_entries}, h={total_g_entries}")


if __name__ == "__main__":
    test_tpi_exploration_analysis()