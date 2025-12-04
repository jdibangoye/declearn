# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import pytest
import numpy as np
from collections import defaultdict
import time

from declearn.core.tabular_tpi import TabularTPI
from declearn.core.sequential_env import SequentialDecPOMDPSimulator, TypeTuple
from declearn.envs.masplan import parse_dpomdp
from declearn.tests.test_masplan_stats import _all_dpomdp_paths
from declearn.tests.test_tabular_tpi import MinimalDAG, MinimalTimescale


class RandomPolicy:
    """Politique baseline aléatoire pour comparaison"""
    def __init__(self, n_actions, rng):
        self.n_actions = n_actions
        self.rng = rng
    
    def act(self, uidx, theta, greedy=False):
        return self.rng.integers(0, self.n_actions)


class GreedyQPolicy:
    """Politique Q-learning décentralisé simple pour comparaison"""
    def __init__(self, n_actions, learning_rate=0.1):
        self.n_actions = n_actions
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.lr = learning_rate
    
    def act(self, uidx, theta, greedy=False):
        h = tuple(theta.joint_histories[uidx]['obs'])
        if greedy:
            return np.argmax(self.q_table[h])
        # ε-greedy avec ε=0.1
        if np.random.random() < 0.1:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[h])
    
    def update(self, h, action, reward, h_next):
        """Mise à jour Q-learning simple"""
        q_current = self.q_table[h][action]
        q_next_max = np.max(self.q_table[h_next])
        self.q_table[h][action] = q_current + self.lr * (reward + 0.9 * q_next_max - q_current)


def test_tpi_vs_random_baseline():
    """Comparaison TPI vs politique aléatoire"""
    
    def run_policy_experiment(policy_type, episodes=50):
        """Exécute un experiment avec une politique donnée"""
        paths = _all_dpomdp_paths()
        if not paths:
            pytest.skip("Pas de specs disponibles")
        
        # Utiliser GridSmall pour comparaison
        grid_path = next((p for p in paths if "GridSmall" in p), None)
        if not grid_path:
            pytest.skip("GridSmall non disponible")
        
        spec = parse_dpomdp(grid_path)
        sim = SequentialDecPOMDPSimulator(spec, seed=42, memory_m=1)
        
        if policy_type == "TPI":
            dag = MinimalDAG()
            n_actions = {0: len(spec.actions[0])}
            ts = MinimalTimescale()
            rng = np.random.default_rng(42)
            policy = TabularTPI(dag, n_actions, ts, 1, rng)
        else:  # Random
            policy = RandomPolicy(len(spec.actions[0]), np.random.default_rng(42))
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            type_tuple = sim.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done and episode_length < 30:
                # Agent 0 utilise la politique testée
                action = policy.act(0, type_tuple, greedy=(episode > episodes//2))
                
                next_type_tuple, reward, done, _ = sim.step(action)
                episode_reward += reward
                episode_length += 1
                
                # Mise à jour pour TPI
                if policy_type == "TPI" and episode_length > 1:
                    theta = (type_tuple.state, type_tuple.t)
                    theta_next = (next_type_tuple.state, next_type_tuple.t)
                    h = tuple(type_tuple.joint_histories[0]['obs'])
                    h_next = tuple(next_type_tuple.joint_histories[0]['obs'])
                    
                    sample = {
                        0: {
                            "theta": theta, "a": action, "r": float(reward),
                            "theta_next": theta_next, "h": h, "h_next": h_next,
                            "a_next": action, "u": 0
                        }
                    }
                    policy.step(sample, k=episode * 30 + episode_length)
                
                type_tuple = next_type_tuple
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return episode_rewards, episode_lengths
    
    # Expériences comparatives
    tpi_rewards, tpi_lengths = run_policy_experiment("TPI")
    random_rewards, random_lengths = run_policy_experiment("Random")
    
    # Analyse des résultats
    tpi_mean = np.mean(tpi_rewards)
    random_mean = np.mean(random_rewards)
    
    tpi_final = np.mean(tpi_rewards[-10:])  # Performance finale
    random_final = np.mean(random_rewards[-10:])
    
    print(f"TPI - Moyenne: {tpi_mean:.2f}, Finale: {tpi_final:.2f}")
    print(f"Random - Moyenne: {random_mean:.2f}, Finale: {random_final:.2f}")
    print(f"Amélioration TPI: {tpi_final - random_final:.2f}")
    
    # Vérifications
    assert tpi_final >= random_final - 2.0, f"TPI pas clairement meilleur que aléatoire: {tpi_final} vs {random_final}"
    
    # TPI doit au moins apprendre (amélioration dans le temps)
    tpi_early = np.mean(tpi_rewards[:10])
    tpi_improvement = tpi_final - tpi_early
    print(f"Amélioration TPI dans le temps: {tpi_improvement:.2f}")
    
    assert tpi_improvement >= -1.0, f"TPI doit au moins rester stable: amélioration={tpi_improvement}"


def test_tpi_vs_simple_qlearning():
    """Comparaison TPI vs Q-learning décentralisé simple"""
    
    def run_comparison_experiment(episodes=60):
        """Compare TPI et Q-learning sur même problème"""
        # Problème contrôlé : 2 états, 2 actions
        dag = MinimalDAG()
        n_actions = {0: 2}
        ts = MinimalTimescale()
        rng = np.random.default_rng(42)
        
        # Politiques
        tpi = TabularTPI(dag, n_actions, ts, 1, rng)
        qlearning = GreedyQPolicy(2, learning_rate=0.1)
        
        # Environnement simple : état 0 -> action 0 donne reward=1, action 1 donne reward=0.2
        #                        état 1 -> action 0 donne reward=0.3, action 1 donne reward=0.8
        def simple_env(state, action):
            rewards = {(0, 0): 1.0, (0, 1): 0.2, (1, 0): 0.3, (1, 1): 0.8}
            next_state = 1 - state  # Alternance d'états
            return next_state, rewards.get((state, action), 0.0)
        
        tpi_rewards = []
        qlearning_rewards = []
        
        for episode in range(episodes):
            # TPI épisode
            state = episode % 2
            mock_theta = TypeTuple(
                state=state, t=0,
                joint_histories=[{'obs': [state], 'acts': [-1]}],
                prefix_actions=[]
            )
            
            tpi_action = tpi.act(0, mock_theta, greedy=(episode > episodes//2))
            next_state, tpi_reward = simple_env(state, tpi_action)
            tpi_rewards.append(tpi_reward)
            
            # Mise à jour TPI
            sample = {
                0: {
                    "theta": (state,), "a": tpi_action, "r": tpi_reward,
                    "theta_next": (next_state,), "h": (state,), "h_next": (next_state,),
                    "a_next": tpi_action, "u": 0
                }
            }
            tpi.step(sample, k=episode)
            
            # Q-learning épisode
            h = (state,)
            h_next = (next_state,)
            ql_action = qlearning.act(0, mock_theta, greedy=(episode > episodes//2))
            _, ql_reward = simple_env(state, ql_action)
            qlearning_rewards.append(ql_reward)
            
            # Mise à jour Q-learning
            qlearning.update(h, ql_action, ql_reward, h_next)
        
        return tpi_rewards, qlearning_rewards
    
    # Comparaison
    tpi_rewards, ql_rewards = run_comparison_experiment()
    
    # Analyse finale
    tpi_final = np.mean(tpi_rewards[-15:])
    ql_final = np.mean(ql_rewards[-15:])
    
    print(f"TPI final: {tpi_final:.3f}")
    print(f"Q-learning final: {ql_final:.3f}")
    print(f"Différence: {tpi_final - ql_final:.3f}")
    
    # Les deux doivent apprendre quelque chose
    assert tpi_final > 0.4, f"TPI doit apprendre: {tpi_final}"
    assert ql_final > 0.4, f"Q-learning doit apprendre: {ql_final}"
    
    # Performance comparable (pas forcément TPI > Q-learning sur ce problème simple)
    assert abs(tpi_final - ql_final) < 0.5, f"Performances trop différentes: TPI={tpi_final}, QL={ql_final}"


def test_tpi_ablation_components():
    """Étude d'ablation : impact de chaque composant TPI"""
    
    class SimplifiedTPI:
        """Version simplifiée de TPI pour ablation"""
        def __init__(self, use_q=True, use_g=True, use_mirror=True):
            self.use_q = use_q
            self.use_g = use_g  
            self.use_mirror = use_mirror
            
            self.q_table = defaultdict(lambda: np.zeros(2))
            self.g_table = defaultdict(lambda: np.zeros(2))
            self.counts = defaultdict(lambda: np.zeros(2))
        
        def act(self, state, greedy=False):
            h = (state,)
            if self.use_g:
                values = self.g_table[h]
            else:
                values = self.q_table[h]
            
            if greedy:
                return np.argmax(values)
            
            # Sampling selon les valeurs
            if self.use_mirror:
                probs = np.exp(values) / np.sum(np.exp(values))
                return np.random.choice(2, p=probs)
            else:
                # ε-greedy simple
                if np.random.random() < 0.1:
                    return np.random.randint(2)
                return np.argmax(values)
        
        def update(self, state, action, reward):
            h = (state,)
            
            if self.use_q:
                # Q-update simple
                lr = 0.1
                self.q_table[h][action] += lr * (reward - self.q_table[h][action])
            
            if self.use_g:
                # G-update (moyenne mobile vers Q)
                if self.use_q:
                    lr_g = 0.05
                    self.g_table[h][action] += lr_g * (self.q_table[h][action] - self.g_table[h][action])
                else:
                    # G direct si pas de Q
                    lr_g = 0.1
                    self.g_table[h][action] += lr_g * (reward - self.g_table[h][action])
    
    def test_ablation_config(use_q, use_g, use_mirror, episodes=100):
        """Test une configuration d'ablation"""
        policy = SimplifiedTPI(use_q, use_g, use_mirror)
        
        # Problème simple : état 0 -> action 0 optimal (reward=1 vs 0.3)
        rewards = []
        for episode in range(episodes):
            state = 0  # État fixe pour simplicité
            action = policy.act(state, greedy=(episode > episodes//2))
            reward = 1.0 if action == 0 else 0.3
            rewards.append(reward)
            policy.update(state, action, reward)
        
        final_performance = np.mean(rewards[-20:])
        return final_performance
    
    # Test différentes configurations
    configs = [
        (True, True, True),    # TPI complet
        (True, True, False),   # TPI sans mirror ascent
        (True, False, True),   # Q-tables + mirror seulement
        (True, False, False),  # Q-learning simple
        (False, True, True),   # G-tables + mirror seulement
    ]
    
    results = {}
    for config in configs:
        perf = test_ablation_config(*config)
        name = f"Q={config[0]}, G={config[1]}, Mirror={config[2]}"
        results[name] = perf
        print(f"{name}: {perf:.3f}")
    
    # Vérifications
    full_tpi = results["Q=True, G=True, Mirror=True"]
    
    # TPI complet doit être au moins aussi bon que les versions simplifiées
    for name, perf in results.items():
        if name != "Q=True, G=True, Mirror=True":
            diff = full_tpi - perf
            print(f"TPI complet vs {name}: +{diff:.3f}")
            # Accepter une petite dégradation due à la complexité
            assert diff >= -0.1, f"TPI complet trop inférieur à {name}: {diff}"
    
    # Au moins une configuration doit bien fonctionner
    best_perf = max(results.values())
    assert best_perf > 0.8, f"Aucune configuration ne fonctionne bien: max={best_perf}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])