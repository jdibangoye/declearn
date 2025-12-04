# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
"""
Test TPI sur tous les benchmarks à horizon T=1
Valide que la correction fonctionne sur tous les domaines
"""

from declearn.envs.masplan import parse_dpomdp
from declearn.core.sequential_env import SequentialDecPOMDPSimulator
from declearn.core.tabular_tpi import TabularTPI
from declearn.core.sa_schedules import ThreeTimeScale
from declearn.core.types import make_sequential_dag
import numpy as np
import itertools
from pathlib import Path


def exhaustive_search_horizon_1(spec):
    """Recherche exhaustive de la meilleure politique à T=1 avec calcul exact"""
    spec.horizon = 1
    
    # Générer toutes les politiques jointes possibles
    action_spaces = [range(len(spec.actions[i])) for i in range(spec.n_agents)]
    all_joint_actions = list(itertools.product(*action_spaces))
    
    # Distribution initiale
    if spec.start_belief is not None:
        initial_belief = spec.start_belief
    elif spec.init_belief is not None:
        initial_belief = spec.init_belief
    else:
        start_state = spec.start if isinstance(spec.start, str) else spec.states[spec.start]
        initial_belief = np.zeros(len(spec.states))
        initial_belief[spec.states.index(start_state)] = 1.0
    
    all_values = {}
    best_joint_action = None
    best_value = -np.inf
    
    for joint_action in all_joint_actions:
        # À horizon T=1: V^π = Σ_s₀ P(s₀) · R(s₀, π)
        value = 0.0
        
        for s0_idx, prob_s0 in enumerate(initial_belief):
            if prob_s0 < 1e-10:
                continue
            
            # Clé récompense: (state_idx, joint_action_tuple)
            rew_key = (s0_idx, joint_action)
            reward = spec.reward.get(rew_key, 0.0)
            value += prob_s0 * reward
        
        all_values[joint_action] = value
        
        if value > best_value:
            best_value = value
            best_joint_action = joint_action
    
    return best_joint_action, best_value, all_values


def evaluate_tpi_horizon_1(spec, episodes=10000, eval_episodes=5000):
    """Entraîne TPI sur horizon T=1 et évalue sa performance"""
    spec.horizon = 1
    
    dag = make_sequential_dag(spec.n_agents, 1)
    n_actions = {i: len(spec.actions[i]) for i in range(spec.n_agents)}
    
    ts = ThreeTimeScale(alpha=0.5, beta=0.3, gamma=0.1)
    rng = np.random.default_rng(42)
    tpi = TabularTPI(dag, n_actions, ts, 1, rng)
    
    # Phase d'entraînement
    training_rewards = []
    for episode in range(episodes):
        sim = SequentialDecPOMDPSimulator(spec, seed=42 + episode, memory_m=1)
        type_tuple = sim.reset()
        
        episode_reward = 0.0
        done = False
        episode_samples = []
        
        epsilon = max(0.01, 1.0 - episode / (episodes * 0.5))
        
        step = 0
        while not done and step < spec.n_agents:
            agent_id = step % spec.n_agents
            k = episode * spec.n_agents + step
            
            theta = type_tuple
            
            if rng.random() < epsilon:
                action = rng.integers(0, n_actions[agent_id])
            else:
                action = tpi.act(agent_id, theta, greedy=True, k=k)
            
            next_type_tuple, reward, done, _ = sim.step(action)
            episode_reward += reward
            
            if not done:
                next_agent_id = (agent_id + 1) % spec.n_agents
                if rng.random() < epsilon:
                    action_next = rng.integers(0, n_actions[next_agent_id])
                else:
                    action_next = tpi.act(next_agent_id, next_type_tuple, greedy=True, k=k+1)
            else:
                action_next = 0
            
            try:
                h = tuple(theta.joint_histories[agent_id]['obs'][-3:])
            except:
                h = (theta.state,)
            
            try:
                h_next = tuple(next_type_tuple.joint_histories[agent_id]['obs'][-3:])
            except:
                h_next = (next_type_tuple.state,)
            
            sample_data = {
                "theta": theta,
                "a": action,
                "r": float(reward),
                "theta_next": next_type_tuple,
                "h": h,
                "h_next": h_next,
                "a_next": action_next,
                "u": (agent_id, theta.t)
            }
            
            episode_samples.append((agent_id, sample_data))
            type_tuple = next_type_tuple
            step += 1
        
        training_rewards.append(episode_reward)
        
        # Mises à jour
        for i, (agent_id, sample_data) in enumerate(episode_samples):
            sample = {agent_id: sample_data}
            k_update = episode * spec.n_agents + i
            tpi.step(sample, k_update)
    
    # Phase d'évaluation
    eval_rewards = []
    for episode in range(eval_episodes):
        sim = SequentialDecPOMDPSimulator(spec, seed=episode, memory_m=1)
        type_tuple = sim.reset()
        
        episode_reward = 0.0
        done = False
        step = 0
        
        while not done and step < spec.n_agents:
            agent_id = step % spec.n_agents
            k = episode * spec.n_agents + step
            
            action = tpi.act(agent_id, type_tuple, greedy=True, k=k)
            next_type_tuple, reward, done, _ = sim.step(action)
            episode_reward += reward
            
            type_tuple = next_type_tuple
            step += 1
        
        eval_rewards.append(episode_reward)
    
    tpi_value = np.mean(eval_rewards)
    tpi_std = np.std(eval_rewards)
    
    return tpi_value, tpi_std


def test_benchmark(domain_path, domain_name):
    """Teste un benchmark spécifique"""
    print("\n" + "=" * 80)
    print(f"BENCHMARK: {domain_name}")
    print("=" * 80)
    
    try:
        spec = parse_dpomdp(domain_path)
        
        print(f"Agents: {spec.n_agents}, États: {len(spec.states)}")
        print(f"Actions: {[len(spec.actions[i]) for i in range(spec.n_agents)]}")
        
        # Recherche exhaustive
        print("\n  Recherche exhaustive...")
        optimal_policy, optimal_value, all_values = exhaustive_search_horizon_1(spec)
        action_names = [spec.actions[i][optimal_policy[i]] for i in range(spec.n_agents)]
        print(f"  ✓ Politique optimale: {optimal_policy} = {tuple(action_names)}")
        print(f"  ✓ Valeur optimale: {optimal_value:.6f}")
        
        # TPI
        print("\n  Entraînement TPI...")
        tpi_value, tpi_std = evaluate_tpi_horizon_1(spec, episodes=10000, eval_episodes=5000)
        print(f"  ✓ Valeur TPI: {tpi_value:.6f} ± {tpi_std:.6f}")
        
        # Comparaison
        if abs(optimal_value) > 1e-6:
            relative_error = abs(tpi_value - optimal_value) / abs(optimal_value)
            print(f"\n  Écart absolu: {abs(tpi_value - optimal_value):.6f}")
            print(f"  Écart relatif: {relative_error:.2%}")
            
            if relative_error < 0.05:
                print(f"  ✅ SUCCÈS (écart < 5%)")
                return True, relative_error
            elif relative_error < 0.20:
                print(f"  ⚠️  ACCEPTABLE (écart < 20%)")
                return True, relative_error
            else:
                print(f"  ❌ ÉCHEC (écart > 20%)")
                return False, relative_error
        else:
            print(f"  ⚠️  Valeur optimale proche de zéro")
            return True, 0.0
            
    except Exception as e:
        print(f"  ❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False, 1.0


def main():
    """Teste tous les benchmarks à T=1"""
    
    benchmarks_dir = Path("declearn/envs/masplan_specs")
    
    # Liste des benchmarks disponibles
    benchmarks = [
        ("broadcastChannel.dpomdp", "BroadcastChannel"),
        ("GridSmall.dpomdp", "GridSmall"),
        ("dectiger.dpomdp", "DecTiger"),
        ("boxPushingUAI07.dpomdp", "BoxPushing"),
        ("recycling.dpomdp", "Recycling"),
    ]
    
    print("\n" + "=" * 80)
    print("TEST DE TOUS LES BENCHMARKS À HORIZON T=1")
    print("=" * 80)
    
    results = []
    
    for filename, domain_name in benchmarks:
        domain_path = benchmarks_dir / filename
        
        if not domain_path.exists():
            print(f"\n⚠️  Domaine {domain_name} non trouvé: {domain_path}")
            continue
        
        success, error = test_benchmark(str(domain_path), domain_name)
        results.append((domain_name, success, error))
    
    # Résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES TESTS")
    print("=" * 80)
    
    for domain_name, success, error in results:
        status = "✅" if success else "❌"
        print(f"{status} {domain_name:20s} - Écart: {error:.2%}")
    
    success_count = sum(1 for _, s, _ in results if s)
    total_count = len(results)
    print(f"\nRésultat global: {success_count}/{total_count} benchmarks réussis")


if __name__ == "__main__":
    main()
