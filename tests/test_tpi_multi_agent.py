# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import numpy as np
from declearn.core.tabular_tpi import TabularTPI
from declearn.tests.test_tabular_tpi import MinimalDAG
from declearn.core.sa_schedules import ThreeTimeScale
from declearn.core.sequential_env import TypeTuple

def test_multi_agent_tpi():
    """Test TPI avec plusieurs agents interagissant"""
    
    # Configuration multi-agent : 2 agents avec actions différentes
    dag = MinimalDAG(2)  # 2 agents
    n_actions = {0: 2, 1: 3}  # Agent 0: 2 actions, Agent 1: 3 actions
    ts = ThreeTimeScale(alpha=1.0, beta=0.1, gamma=0.01)
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(dag, n_actions, ts, memory_m=1, rng=rng)
    
    print("=== TEST MULTI-AGENT ===")
    print(f"Agents configurés: {list(tpi.policies.keys())}")
    print(f"Actions par agent: {n_actions}")
    
    # Sample joint avec 2 agents
    sample_multi = {
        0: {
            "theta": TypeTuple(
                state=1, t=0, 
                joint_histories=[
                    {'obs': [1], 'acts': [-1]},  # Agent 0
                    {'obs': [2], 'acts': [-1]}   # Agent 1
                ], 
                prefix_actions=[]
            ),
            "a": 1, "r": 1.0,
            "theta_next": TypeTuple(
                state=2, t=1,
                joint_histories=[
                    {'obs': [1,2], 'acts': [-1,1]},  # Agent 0
                    {'obs': [2,3], 'acts': [-1,0]}   # Agent 1
                ],
                prefix_actions=[]
            ),
            "h": (1,), "u": 0
        },
        1: {
            "theta": TypeTuple(
                state=1, t=0,
                joint_histories=[
                    {'obs': [1], 'acts': [-1]},  # Agent 0  
                    {'obs': [2], 'acts': [-1]}   # Agent 1
                ],
                prefix_actions=[]
            ),
            "a": 2, "r": 0.5,
            "theta_next": TypeTuple(
                state=2, t=1,
                joint_histories=[
                    {'obs': [1,2], 'acts': [-1,1]},  # Agent 0
                    {'obs': [2,3], 'acts': [-1,0]}   # Agent 1
                ],
                prefix_actions=[]
            ),
            "h": (2,), "u": 1
        }
    }
    
    # Mise à jour multi-agent
    tpi.step(sample_multi, k=0)
    
    print(f"\n=== RÉSULTATS MULTI-AGENT ===")
    g_stats = tpi.get_g_table_stats()
    print(f"G-tables créées: {list(g_stats.keys())}")
    
    # Compter Q-tables par sous-paire u=(agent, t)
    agent_0_entries = sum(len(q_table) for u_key, q_table in tpi.q_tables.items() if u_key[0] == 0)
    agent_1_entries = sum(len(q_table) for u_key, q_table in tpi.q_tables.items() if u_key[0] == 1)
    print(f"Q-tables agent 0: {agent_0_entries} états")
    print(f"Q-tables agent 1: {agent_1_entries} états")
    
    # Vérifier actions de chaque agent
    mock_theta = TypeTuple(state=1, t=0, joint_histories=[{'obs': [1]}, {'obs': [2]}], prefix_actions=[])
    action_0 = tpi.act(0, mock_theta, greedy=True)
    action_1 = tpi.act(1, mock_theta, greedy=True)
    
    print(f"Action agent 0: {action_0} (max={n_actions[0]-1})")
    print(f"Action agent 1: {action_1} (max={n_actions[1]-1})")

if __name__ == "__main__":
    test_multi_agent_tpi()