# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import pytest
import numpy as np
from declearn.core.tabular_tpi import TabularTPI
from declearn.core.sequential_env import TypeTuple
from declearn.tests.test_tabular_tpi import MinimalDAG, MinimalTimescale


def test_tpi_stochastic_environments():
    """Test TPI sur environnements stochastiques"""
    
    dag = MinimalDAG()
    n_actions = {0: 2}
    ts = MinimalTimescale()
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(dag, n_actions, ts, 1, rng)
    
    # Environnement stochastique : rewards variables
    episode_rewards = []
    
    for episode in range(200):
        action = episode % 2
        
        # Reward stochastique : action 0 meilleure en moyenne
        if action == 0:
            reward = np.random.normal(0.8, 0.2)  # Moyenne 0.8, variance 0.2
        else:
            reward = np.random.normal(0.4, 0.2)  # Moyenne 0.4, variance 0.2
        
        episode_rewards.append(reward)
        
        sample = {
            0: {
                "theta": (1,), "a": action, "r": reward, "theta_next": (1,),
                "h": (1,), "h_next": (1,), "a_next": action, "u": 0
            }
        }
        tpi.step(sample, k=episode)
    
    # Test politique finale sur environnement stochastique
    mock_theta = TypeTuple(
        state=1, t=0,
        joint_histories=[{'obs': [1], 'acts': [-1]}],
        prefix_actions=[]
    )
    
    # Statistiques sur actions choisies
    greedy_actions = [tpi.act(0, mock_theta, greedy=True) for _ in range(20)]
    sampling_actions = [tpi.act(0, mock_theta, greedy=False) for _ in range(100)]
    
    # Vérifications robustesse
    greedy_action = greedy_actions[0]
    assert all(a == greedy_action for a in greedy_actions), "Politique greedy non déterministe"
    
    # Dans un environnement stochastique, TPI doit quand même apprendre la tendance
    action_0_freq = sampling_actions.count(0) / len(sampling_actions)
    print(f"Fréquence action 0 (meilleure en moyenne): {action_0_freq:.2f}")
    
    # Action 0 doit être préférée (mais pas forcément 100% du temps en stochastique)
    assert greedy_action == 0, f"TPI devrait préférer action 0 en greedy: {greedy_action}"
    assert action_0_freq > 0.3, f"Action 0 trop peu échantillonnée: {action_0_freq}"


def test_tpi_partial_observability_levels():
    """Test TPI avec différents niveaux d'observabilité partielle"""
    
    def test_observability_level(history_noise_level):
        """Test avec un niveau de bruit dans les historiques"""
        dag = MinimalDAG()
        n_actions = {0: 2}
        ts = MinimalTimescale()
        rng = np.random.default_rng(42)
        
        tpi = TabularTPI(dag, n_actions, ts, 1, rng)
        
        # Simuler observabilité partielle via historiques bruités
        final_rewards = []
        
        for episode in range(100):
            # État vrai
            true_state = episode % 3
            
            # Observation bruitée selon le niveau
            if np.random.random() < history_noise_level:
                # Observation incorrecte
                observed_state = (true_state + 1) % 3
            else:
                # Observation correcte
                observed_state = true_state
            
            # Action selon politique TPI
            mock_theta = TypeTuple(
                state=observed_state, t=0,
                joint_histories=[{'obs': [observed_state], 'acts': [-1]}],
                prefix_actions=[]
            )
            
            action = tpi.act(0, mock_theta, greedy=(episode > 50))
            
            # Reward basé sur l'état vrai (pas l'observation)
            if true_state == 0:
                reward = 1.0 if action == 0 else 0.2
            elif true_state == 1:
                reward = 0.8 if action == 1 else 0.3
            else:  # true_state == 2
                reward = 0.6 if action == 0 else 0.7
            
            final_rewards.append(reward)
            
            # Sample TPI basé sur observation
            sample = {
                0: {
                    "theta": (observed_state,), "a": action, "r": reward,
                    "theta_next": (observed_state,), "h": (observed_state,),
                    "h_next": (observed_state,), "a_next": action, "u": 0
                }
            }
            tpi.step(sample, k=episode)
        
        return np.mean(final_rewards[-20:])  # Performance finale
    
    # Test différents niveaux de bruit
    noise_levels = [0.0, 0.2, 0.5, 0.8]
    performances = {}
    
    for noise in noise_levels:
        perf = test_observability_level(noise)
        performances[noise] = perf
        print(f"Bruit {noise:.1f}: performance {perf:.3f}")
    
    # Vérifications
    # Plus de bruit doit dégrader la performance
    assert performances[0.0] >= performances[0.5] - 0.1, "Performance doit se dégrader avec le bruit"
    assert performances[0.5] >= performances[0.8] - 0.1, "Performance doit se dégrader avec plus de bruit"
    
    # Même avec beaucoup de bruit, TPI doit apprendre quelque chose
    assert performances[0.8] > 0.3, f"Performance trop faible même avec bruit élevé: {performances[0.8]}"


def test_tpi_initialization_sensitivity():
    """Test sensibilité aux conditions initiales"""
    
    def run_with_seed(seed):
        """Exécute TPI avec une graine spécifique"""
        from declearn.core.sequential_env import TypeTuple
        
        dag = MinimalDAG()
        n_actions = {0: 2}
        ts = MinimalTimescale()
        rng = np.random.default_rng(seed)
        
        tpi = TabularTPI(dag, n_actions, ts, 1, rng)
        
        # Créer des TypeTuple corrects
        theta = TypeTuple(
            state=1,
            t=0,
            joint_histories=[{"obs": [1], "acts": []}],
            prefix_actions=[]
        )
        theta_next = TypeTuple(
            state=1,
            t=1,
            joint_histories=[{"obs": [1], "acts": [0]}],
            prefix_actions=[]
        )
        
        # Problème déterministe pour isoler l'effet de l'initialisation
        for episode in range(100):
            action = episode % 2
            reward = 1.0 if action == 0 else 0.3
            
            sample = {
                0: {
                    "theta": theta, "a": action, "r": reward, "theta_next": theta_next,
                    "h": (1,), "h_next": (1,), "a_next": action, "u": 0
                }
            }
            tpi.step(sample, k=episode)
        
        # Performance finale
        mock_theta = TypeTuple(
            state=1, t=0,
            joint_histories=[{'obs': [1], 'acts': [-1]}],
            prefix_actions=[]
        )
        
        # Test sur plusieurs échantillons pour moyenner
        actions = [tpi.act(0, mock_theta, greedy=True) for _ in range(10)]
        final_action = max(set(actions), key=actions.count)  # Action la plus fréquente
        
        # Utiliser nouvelle structure u=(agent, t)
        u_pair = (0, 0)  # agent=0, t=0
        q_values = None
        if u_pair in tpi.q_tables and theta in tpi.q_tables[u_pair]:
            q_values = tpi.q_tables[u_pair][theta].copy()
        
        return final_action, q_values
    
    # Test avec différentes graines
    seeds = [42, 123, 456, 789, 999]
    results = {}
    
    for seed in seeds:
        action, q_values = run_with_seed(seed)
        results[seed] = {'action': action, 'q_values': q_values}
        print(f"Seed {seed}: action finale={action}, Q-values={q_values}")
    
    # Vérifications robustesse
    final_actions = [results[seed]['action'] for seed in seeds]
    
    # Toutes les graines doivent converger vers la même action optimale
    optimal_action = max(set(final_actions), key=final_actions.count)
    convergence_rate = final_actions.count(optimal_action) / len(final_actions)
    
    print(f"Action optimale: {optimal_action}, convergence: {convergence_rate:.2f}")
    
    assert convergence_rate >= 0.8, f"Convergence insuffisante entre graines: {convergence_rate}"
    assert optimal_action == 0, f"Action optimale incorrecte: {optimal_action}"
    
    # Q-values doivent être cohérentes entre graines
    q_values_list = [results[seed]['q_values'] for seed in seeds if results[seed]['q_values'] is not None]
    
    if len(q_values_list) > 1:
        q_diff_variance = np.var([q[0] - q[1] for q in q_values_list])
        print(f"Variance différence Q entre graines: {q_diff_variance:.4f}")
        
        # Faible variance = robustesse
        assert q_diff_variance < 0.1, f"Q-values trop variables entre graines: {q_diff_variance}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])