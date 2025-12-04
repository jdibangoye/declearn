# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
"""Test pour vérifier la structure Q-tables vs théorie TPI"""

import pytest
import numpy as np
from declearn.core.tabular_tpi import TabularTPI
from declearn.tests.test_tabular_tpi import MinimalDAG, MinimalTimescale
from declearn.core.sequential_env import TypeTuple


def test_tpi_structure_analysis():
    """Analyser la structure Q-tables vs théorie TPI u=(i,t)"""
    
    print("\n=== ANALYSE STRUCTURE TPI ===")
    
    # Configuration multi-agent, multi-time
    n_agents = 2
    horizon = 3
    dag = MinimalDAG(n_agents)
    n_actions = {0: 2, 1: 3}  # Agent 0: 2 actions, Agent 1: 3 actions
    ts = MinimalTimescale()
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(dag, n_actions, ts, 1, rng)
    
    print(f"Configuration:")
    print(f"  - Agents: {n_agents}")
    print(f"  - Horizon: {horizon}")
    print(f"  - Actions par agent: {n_actions}")
    
    print(f"\nStructure TPI actuelle:")
    print(f"  - Q-tables keys: {list(tpi.q_tables.keys())}")
    print(f"  - G-tables keys: {list(tpi.g_tables.keys())}")
    print(f"  - Nombre Q-tables: {len(tpi.q_tables)}")
    print(f"  - Nombre G-tables: {len(tpi.g_tables)}")
    
    # Théorie TPI : devrait avoir T×N sous-paires u=(i,t)
    expected_u_pairs = [(i, t) for i in range(n_agents) for t in range(horizon)]
    print(f"\nThéorie TPI - Sous-paires u=(i,t) attendues:")
    print(f"  - {expected_u_pairs}")
    print(f"  - Nombre attendu: {len(expected_u_pairs)}")
    
    # Simuler quelques types θ et voir comment ils sont stockés
    print(f"\nTest avec types θ différents:")
    
    sample_types = []
    for t in range(horizon):
        for state in range(2):  # 2 états
            theta = TypeTuple(
                state=state,
                t=t,
                joint_histories=[{'obs': [state], 'acts': [-1]} for _ in range(n_agents)],
                prefix_actions=[]
            )
            sample_types.append((t, state, theta))
    
    # Créer des samples pour chaque agent et type
    for agent_id in range(n_agents):
        print(f"\n  Agent {agent_id}:")
        for t, state, theta in sample_types[:4]:  # Premiers quelques types
            sample = {
                agent_id: {
                    "theta": theta,
                    "a": 0,
                    "r": 1.0,
                    "theta_next": theta,  # Simplifié
                    "h": (state,),
                    "h_next": (state,),
                    "a_next": 0,
                    "u": agent_id
                }
            }
            
            tpi.step(sample, k=t)
            print(f"    θ(s={state},t={t}) ajouté à Q-table[{agent_id}]")
    
    # Analyser résultat
    print(f"\nRésultat après ajouts:")
    for agent_id, q_table in tpi.q_tables.items():
        print(f"  Q-table[{agent_id}]: {len(q_table)} types θ")
        if len(q_table) > 0:
            # Examiner les clés
            sample_keys = list(q_table.keys())[:3]
            print(f"    Exemples clés: {sample_keys}")
    
    print(f"  G-tables: {len(tpi.g_tables)} sous-paires")
    for u_pair, g_table in tpi.g_tables.items():
        print(f"    G-table[{u_pair}]: {len(g_table)} historiques h")
    
    # Conclusion
    total_q_entries = sum(len(q_table) for q_table in tpi.q_tables.values())
    
    print(f"\n=== CONCLUSION ===")
    print(f"Structure actuelle:")
    print(f"  - {len(tpi.q_tables)} Q-tables (par agent)")
    print(f"  - {len(tpi.g_tables)} G-tables (par sous-paire u=(i,t))")
    print(f"  - {total_q_entries} types θ stockés (total)")
    
    print(f"\nThéorie TPI attendue:")
    print(f"  - {len(expected_u_pairs)} Q-tables (par sous-paire u=(i,t))")
    print(f"  - {len(expected_u_pairs)} G-tables (par sous-paire u=(i,t))")
    
    # Diagnostic du problème
    if len(tpi.q_tables) == n_agents:
        print(f"\n⚠️  PROBLÈME IDENTIFIÉ:")
        print(f"    Q-tables indexées par AGENT (actuel) vs u=(i,t) (théorie)")
        print(f"    Cela explique pourquoi on voit Q={n_agents} au lieu de Q={len(expected_u_pairs)}")
    
    return {
        'q_tables_count': len(tpi.q_tables),
        'g_tables_count': len(tpi.g_tables),
        'expected_u_pairs': len(expected_u_pairs),
        'total_theta_types': total_q_entries
    }


if __name__ == "__main__":
    result = test_tpi_structure_analysis()
    print(f"\nRésultats: {result}")