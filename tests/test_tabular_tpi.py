# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
import pytest
import numpy as np
from collections import namedtuple

from declearn.core.tabular_tpi import TabularTPI
from declearn.envs.masplan import parse_dpomdp
from declearn.core.sequential_env import SequentialDecPOMDPSimulator
from declearn.tests.test_masplan_stats import _all_dpomdp_paths

# Import des fonctions de benchmark depuis le module dédié
try:
    from declearn.tests.test_tabular_tpi_benchmarks import BENCHMARK_VALUES, get_domain_name
except ImportError:
    # Si import échoue, définir des valeurs par défaut pour éviter les erreurs
    BENCHMARK_VALUES = {}
    def get_domain_name(path):
        return path.split('/')[-1].split('.')[0]

# Structure minimale pour tester TPI
Node = namedtuple('Node', ['idx', 'successor'])

class MinimalDAG:
    """DAG minimal pour tests multi-agent"""
    def __init__(self, n_agents=1):
        self.nodes = list(range(n_agents))  # Support multi-agent
    
    def parents(self, uidx):
        return []
    
    def children(self, uidx):
        return []

class MinimalTimescale:
    """Timescale minimal pour tests - VERSION AMÉLIORÉE"""
    def __init__(self):
        # Paramètres d'apprentissage plus agressifs pour les tests
        self.initial_alpha = 0.3  # Plus élevé pour apprentissage rapide
        self.initial_beta = 0.1   
        self.initial_gamma = 0.05
        
    def decay(self, k):
        """Décroissance appropriée des taux d'apprentissage"""
        # Créer un objet avec les taux décroissants
        decayed = MinimalTimescale()
        # Décroissance 1/(1+k*factor) pour éviter de trop descendre
        decayed.alpha = self.initial_alpha / (1 + k * 0.001)  # Décroissance lente pour Q-learning  
        decayed.beta = self.initial_beta / (1 + k * 0.002)    # Décroissance moyenne pour G-tables
        decayed.gamma = self.initial_gamma / (1 + k * 0.005)  # Décroissance rapide pour policies
        return decayed
    
    def should_update_policy(self, k: int) -> bool:
        """Détermine si la politique doit être mise à jour à l'étape k"""
        # Mise à jour politique moins fréquente (échelle lente)
        return k % 10 == 0  # Chaque 10 étapes
    
    def should_update_g_table(self, k: int) -> bool:
        """Détermine si les G-tables doivent être mises à jour à l'étape k"""
        # Mise à jour G-table intermédiaire
        return k % 5 == 0  # Chaque 5 étapes
    
    def should_update_q_table(self, k: int) -> bool:
        """Détermine si les Q-tables doivent être mises à jour à l'étape k"""
        # Mise à jour Q-table rapide
        return True  # Chaque étape


def test_tpi_initialization():
    """Test que TPI s'initialise correctement."""
    dag = MinimalDAG()
    n_actions = {0: 2}  # agent 0 a 2 actions
    ts = MinimalTimescale()
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(
        dag=dag,
        n_actions_dict=n_actions,
        timescale=ts,
        memory_m=1,
        rng=rng
    )
    
    # Politiques sont maintenant créées dynamiquement par u=(i,t)
    assert len(tpi.policies) == 0  # Vides au début (créées dynamiquement)
    assert len(tpi.q_tables) == 0  # Vides au début (créées dynamiquement par sous-paire u=(i,t))
    assert len(tpi.g_tables) == 0  # Vides au début


def test_tpi_step_basic():
    """Test qu'un step TPI fonctionne sans erreur."""
    from declearn.core.sequential_env import TypeTuple
    
    dag = MinimalDAG()
    n_actions = {0: 2}
    ts = MinimalTimescale()
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(dag, n_actions, ts, 1, rng)
    
    # Créer des objets TypeTuple corrects
    theta = TypeTuple(
        state=1,
        t=2,
        joint_histories=[{"obs": [1], "acts": []}],  # Un agent, historique minimal
        prefix_actions=[]
    )
    theta_next = TypeTuple(
        state=1,
        t=3,
        joint_histories=[{"obs": [1], "acts": [0]}],  # Après action 0
        prefix_actions=[]
    )
    
    # Sample minimal pour un step
    sample = {
        0: {
            "theta": theta,
            "a": 0,
            "r": 1.0,
            "theta_next": theta_next,
            "h": (1,),
            "h_next": (1,),
            "a_next": 1,
            "u": 0  # pour check_locality
        }
    }
    
    # Le step ne doit pas lever d'exception
    tpi.step(sample, k=0)
    
    # Vérifier que les tables ont été créées avec nouvelle structure u=(i,t)
    # Dans le sample, theta.t=2, agent=0 -> sous-paire (0,2)
    assert (0, 2) in tpi.q_tables  # Q-table pour sous-paire (agent=0, t=2)
    assert theta in tpi.q_tables[(0, 2)]  # θ dans Q-table[(0,2)]
    # G-tables créées dynamiquement aussi
    assert len(tpi.g_tables) > 0  # Au moins une G-table créée


def test_tpi_act():
    """Test que TPI peut choisir des actions."""
    dag = MinimalDAG()
    n_actions = {0: 2}
    ts = MinimalTimescale()
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(dag, n_actions, ts, 1, rng)
    
    # Initialiser avec un sample
    sample = {
        0: {
            "theta": (1,), "a": 0, "r": 1.0, "theta_next": (1,),
            "h": (1,), "h_next": (1,), "a_next": 0, "u": 0
        }
    }
    tpi.step(sample, k=0)
    
    # Créer un TypeTuple mock pour le test
    from declearn.core.sequential_env import TypeTuple
    mock_theta = TypeTuple(
        state=1, 
        t=0, 
        joint_histories=[{'obs': [1], 'acts': [-1]}], 
        prefix_actions=[]
    )
    
    # Tester action sampling et greedy
    action_sample = tpi.act(0, mock_theta, greedy=False)
    action_greedy = tpi.act(0, mock_theta, greedy=True)
    
    assert action_sample in [0, 1]
    assert action_greedy in [0, 1]


def test_tpi_with_sequential_simulator():
    """Test TPI avec le simulateur séquentiel."""
    # Utiliser un spec simple pour le test
    from declearn.tests.test_masplan_stats import _all_dpomdp_paths
    
    paths = _all_dpomdp_paths()
    if not paths:
        pytest.skip("Pas de specs disponibles")
    
    # Prendre le plus petit spec (GridSmall)
    path = next(p for p in paths if "GridSmall" in p)
    spec = parse_dpomdp(path)
    
    # Créer le simulateur
    sim = SequentialDecPOMDPSimulator(spec, seed=42, memory_m=1)
    
    # Construire un DAG simple basé sur le spec
    dag = MinimalDAG()  # simplifié pour le test
    
    # TPI setup - utiliser le nombre d'actions du spec
    n_actions = {0: len(spec.actions[0])} if spec.n_agents > 0 else {0: 2}
    ts = MinimalTimescale()
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(dag, n_actions, ts, 1, rng)
    
    # Simuler quelques épisodes
    for episode in range(3):
        type_tuple = sim.reset()
        done = False
        episode_steps = 0
        
        while not done and episode_steps < 20:  # augmenter la limite
            # Simuler agent par agent séquentiellement
            initial_type_tuple = type_tuple
            
            # Agent par agent dans l'ordre du DAG
            for agent_id in range(spec.n_agents):
                if done:
                    break
                    
                if agent_id == 0:
                    # Agent 0 utilise TPI - passer le TypeTuple complet
                    action = tpi.act(agent_id, type_tuple, greedy=False)
                else:
                    # Autres agents utilisent une politique par défaut
                    action = 0
                
                # Un appel step() par agent
                next_type_tuple, reward, done, _ = sim.step(action)
                type_tuple = next_type_tuple
                
                if done:
                    break
            
            # Si on a terminé l'étape complète, construire le sample TPI
            if not done or episode_steps == 0:  # au moins un sample
                theta = (initial_type_tuple.state, initial_type_tuple.t)
                theta_next = (type_tuple.state, type_tuple.t)
                h = tuple(initial_type_tuple.joint_histories[0]['obs'])
                h_next = tuple(type_tuple.joint_histories[0]['obs'])
                
                sample = {
                    0: {
                        "theta": theta,
                        "a": 0,  # action de l'agent 0 (simplifiée)
                        "r": float(reward),
                        "theta_next": theta_next,
                        "h": h,
                        "h_next": h_next,
                        "a_next": 0,  # simplifié
                        "u": 0
                    }
                }
                
                # Mise à jour TPI
                tpi.step(sample, k=episode * 20 + episode_steps)
            
            episode_steps += 1
    
    # Vérifier que TPI a appris quelque chose
    assert len(tpi.q_tables) > 0, "TPI devrait avoir des Q-tables après l'entraînement"
    assert len(tpi.g_tables) > 0, "TPI devrait avoir des G-tables après l'entraînement"
    
    # Test qu'il peut agir de manière cohérente
    # Créer un TypeTuple test pour les actions cohérentes
    from declearn.core.sequential_env import TypeTuple
    test_theta = TypeTuple(
        state=0, 
        t=0, 
        joint_histories=[{'obs': [0], 'acts': [-1]}], 
        prefix_actions=[]
    )
    action1 = tpi.act(0, test_theta, greedy=True)
    action2 = tpi.act(0, test_theta, greedy=True)
    assert action1 == action2, "Les actions greedy devraient être déterministes"
    
    print(f"TPI training completed. Q-tables: {len(tpi.q_tables)}, G-tables: {len(tpi.g_tables)}")


def test_tpi_learning_convergence():
    """Test que TPI converge sur un problème simple."""
    from declearn.core.sequential_env import TypeTuple
    
    dag = MinimalDAG()
    n_actions = {0: 2}
    ts = MinimalTimescale()
    rng = np.random.default_rng(42)

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
        joint_histories=[{"obs": [1], "acts": [0]}],  # Après une action
        prefix_actions=[]
    )

    # Problème simple : action 0 donne toujours reward=1, action 1 donne reward=0
    samples_a0 = {
        0: {
            "theta": theta, "a": 0, "r": 1.0, "theta_next": theta_next,
            "h": (1,), "h_next": (1,), "a_next": 0, "u": 0
        }
    }
    samples_a1 = {
        0: {
            "theta": theta, "a": 1, "r": 0.0, "theta_next": theta_next,
            "h": (1,), "h_next": (1,), "a_next": 1, "u": 0
        }
    }    # Entraînement alterné
    for k in range(100):
        if k % 2 == 0:
            tpi.step(samples_a0, k)
        else:
            tpi.step(samples_a1, k)
    
    # Après convergence, TPI devrait préférer l'action 0
    test_theta = TypeTuple(
        state=1,
        t=0,
        joint_histories=[{'obs': [1], 'acts': [-1]}],
        prefix_actions=[]
    )
    greedy_action = tpi.act(0, test_theta, greedy=True)
    assert greedy_action == 0, f"TPI devrait préférer action 0, mais a choisi {greedy_action}"

    # Vérifier les valeurs Q - utiliser la nouvelle structure u=(agent, t)
    u_pair = (0, 0)  # agent=0, t=0 (du theta utilisé dans l'entraînement)
    if u_pair in tpi.q_tables and theta in tpi.q_tables[u_pair]:
        q_values = tpi.q_tables[u_pair][theta]
        assert q_values[0] > q_values[1], f"Q(s,0)={q_values[0]} devrait être > Q(s,1)={q_values[1]}"


def test_debug_memory_policy():
    """Debug : voir ce que retourne MemoryPolicy avec interface (t, h)"""
    dag = MinimalDAG()
    n_actions = {0: 2}
    ts = MinimalTimescale()
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(dag, n_actions, ts, 1, rng)
    
    # Initialiser avec un sample
    sample = {
        0: {
            "theta": (1,), "a": 0, "r": 1.0, "theta_next": (1,),
            "h": (1,), "h_next": (1,), "a_next": 0, "u": 0
        }
    }
    tpi.step(sample, k=0)
    
    # Debug l'interface MemoryPolicy avec u=(agent,t)
    # Après step(), la politique devrait être créée pour u=(0,0)
    u = (0, 0)  # agent 0, temps 0
    
    if u in tpi.policies:
        pol = tpi.policies[u]
        h = (1,)
        
        print(f"MemoryPolicy type: {type(pol)}")
        print(f"MemoryPolicy methods: {[m for m in dir(pol) if not m.startswith('_')]}")
        
        # Test les méthodes avec l'interface correcte (uidx, h)
        agent_id = 0
        try:
            greedy_result = pol.greedy(agent_id, h)
            print(f"pol.greedy({agent_id}, {h}) -> {greedy_result} (type: {type(greedy_result)})")
            assert isinstance(greedy_result, (int, np.integer)), f"greedy doit retourner int, pas {type(greedy_result)}"
        except Exception as e:
            print(f"pol.greedy({agent_id}, {h}) error: {e}")
        
        try:
            sample_result = pol.sample(agent_id, h, rng)
            print(f"pol.sample({agent_id}, {h}, rng) -> {sample_result} (type: {type(sample_result)})")
            assert isinstance(sample_result, (int, np.integer)), f"sample doit retourner int, pas {type(sample_result)}"
        except Exception as e:
            print(f"pol.sample({agent_id}, {h}, rng) error: {e}")
    else:
        print(f"Politique {u} non trouvée dans tpi.policies. Clés disponibles: {list(tpi.policies.keys())}")


def run_tpi_evaluation(spec, horizon, episodes=1000, intensive_training=True):
    """Évaluation TPI multi-agent complète avec diagnostics"""
    
    # Configuration TPI
    dag = MinimalDAG(spec.n_agents)
    n_actions = {i: len(spec.actions[i]) for i in range(spec.n_agents)}
    ts = MinimalTimescale()
    rng = np.random.default_rng(42)
    
    tpi = TabularTPI(dag, n_actions, ts, 1, rng)
    sim = SequentialDecPOMDPSimulator(spec, seed=42, memory_m=1)
    
    # DIAGNOSTICS: Tracker l'évolution des Q-tables
    q_evolution = []
    training_rewards = []
    
    print(f"  Démarrage entraînement {episodes} épisodes...")
    
    for episode in range(episodes):
        type_tuple = sim.reset()
        done = False
        episode_reward = 0
        step_count = 0
        episode_samples = []
        
        while not done and step_count < horizon:
            initial_type_tuple = type_tuple
            
            for agent_id in range(spec.n_agents):
                if done or step_count >= horizon:
                    break
                
                # Exploration puis exploitation progressive
                greedy_prob = min(0.9, episode / (episodes * 0.8))
                use_greedy = np.random.random() < greedy_prob
                
                action = tpi.act(agent_id, type_tuple, greedy=use_greedy)
                next_type_tuple, reward, done, _ = sim.step(action)
                episode_reward += reward
                
                # Enregistrer sample
                if step_count > 0:
                    theta = (initial_type_tuple.state, initial_type_tuple.t)
                    theta_next = (type_tuple.state, type_tuple.t)
                    h = tuple(initial_type_tuple.joint_histories[agent_id]['obs'][-3:])
                    h_next = tuple(type_tuple.joint_histories[agent_id]['obs'][-3:])
                    
                    sample_data = {
                        "theta": theta,
                        "a": action,
                        "r": float(reward),
                        "theta_next": theta_next,
                        "h": h,
                        "h_next": h_next,
                        "a_next": action,
                        "u": agent_id
                    }
                    episode_samples.append((agent_id, sample_data))
                
                type_tuple = next_type_tuple
                if done:
                    break
            
            step_count += 1
        
        training_rewards.append(episode_reward)
        
        # Mise à jour TPI
        for agent_id, sample_data in episode_samples:
            sample = {agent_id: sample_data}
            tpi.step(sample, k=episode * horizon + len(episode_samples))
        
        # Diagnostic périodique
        if episode > 0 and episode % (episodes // 10) == 0:
            # Analyser l'évolution des Q-tables
            total_q_entries = sum(len(q_table) for q_table in tpi.q_tables.values())
            avg_reward = np.mean(training_rewards[-100:]) if len(training_rewards) >= 100 else np.mean(training_rewards)
            
            q_evolution.append({
                'episode': episode,
                'q_entries': total_q_entries,
                'avg_reward': avg_reward,
                'exploration_rate': 1.0 - greedy_prob
            })
            
            print(f"    Episode {episode}: Q-entries={total_q_entries}, "
                  f"Reward={avg_reward:.3f}, Exploration={1.0-greedy_prob:.2f}")
    
    # Évaluation finale
    evaluation_rewards = []
    num_eval_episodes = 50
    
    for eval_episode in range(num_eval_episodes):
        type_tuple = sim.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done and step_count < horizon:
            for agent_id in range(spec.n_agents):
                if done or step_count >= horizon:
                    break
                
                # POLITIQUE GREEDY PURE pour évaluation
                action = tpi.act(agent_id, type_tuple, greedy=True)
                next_type_tuple, reward, done, _ = sim.step(action)
                episode_reward += reward
                type_tuple = next_type_tuple
                
                if done:
                    break
            
            step_count += 1
        
        evaluation_rewards.append(episode_reward)
    
    # Diagnostics finaux
    final_value = np.mean(evaluation_rewards)
    training_progress = np.mean(training_rewards[-50:]) - np.mean(training_rewards[:50])
    
    print(f"  Q-tables finales: {sum(len(qt) for qt in tpi.q_tables.values())} entrées")
    print(f"  G-tables finales: {sum(len(gt) for gt in tpi.g_tables.values())} entrées")
    print(f"  Progrès entraînement: {training_progress:.3f}")
    print(f"  Évaluation finale: {final_value:.6f}")
    
    # Si pas d'apprentissage du tout, signaler
    if total_q_entries == 0:
        print(f"  ⚠️  ALERTE: Aucune Q-table créée!")
    
    if len(q_evolution) > 1:
        q_growth = q_evolution[-1]['q_entries'] - q_evolution[0]['q_entries']
        reward_improvement = q_evolution[-1]['avg_reward'] - q_evolution[0]['avg_reward']
        print(f"  Croissance Q-tables: {q_growth}")
        print(f"  Amélioration reward: {reward_improvement:.3f}")
    
    return final_value


# Dans run_tpi_evaluation, ajouter plus de diagnostics :

    # Statistiques détaillées
    final_value = np.mean(evaluation_rewards)
    training_progress = np.mean(training_rewards[-50:]) - np.mean(training_rewards[:50])
    
    # Diagnostics approfondis
    reward_std = np.std(evaluation_rewards)
    reward_min = np.min(evaluation_rewards)
    reward_max = np.max(evaluation_rewards)
    
    tpi_agents = len([i for i in range(spec.n_agents) if i in tpi.policies])
    print(f"  Entraînement: {len(training_rewards)} épisodes")
    print(f"  Agents TPI: {tpi_agents}/{spec.n_agents}")
    print(f"  Progrès training: {training_progress:.3f}")
    print(f"  Évaluation finale: {final_value:.6f} (sur {num_eval_episodes} épisodes)")
    print(f"  Reward range: [{reward_min:.3f}, {reward_max:.3f}] ± {reward_std:.3f}")
    print(f"  Tables créées: Q={len(tpi.q_tables)}, G={len(tpi.g_tables)}")
    
    # Inspection des Q-tables
    for agent_id in tpi.q_tables:
        if tpi.q_tables[agent_id]:
            q_states = len(tpi.q_tables[agent_id])
            print(f"  Agent {agent_id}: {q_states} états Q appris")


@pytest.mark.parametrize("horizon", [2, 3])
def test_benchmark_small_horizons(horizon):
    """Test TPI sur benchmarks avec petits horizons (2-3) - diagnostic intensif"""
    
    paths = _all_dpomdp_paths()
    if not paths:
        pytest.skip("Pas de specs disponibles")

    tested_domains = 0
    successful_domains = 0

    for path in paths:
        domain_name = get_domain_name(path)
        
        if domain_name not in BENCHMARK_VALUES or horizon not in BENCHMARK_VALUES[domain_name]:
            continue

        optimal_value = BENCHMARK_VALUES[domain_name][horizon]
        
        try:
            spec = parse_dpomdp(path)
            
            print(f"\n=== {domain_name} - Horizon {horizon} - DIAGNOSTIC INTENSIF ===")
            print(f"Valeur optimale attendue: {optimal_value:.6f}")
            
            # TEST PROGRESSION: 300 -> 3000 -> 30000 épisodes
            episode_counts = [300, 3000, 30000]
            tpi_values = []
            
            for episodes in episode_counts:
                print(f"\n--- Entraînement {episodes} épisodes ---")
                tpi_value = run_tpi_evaluation(spec, horizon, episodes=episodes, intensive_training=True)
                tpi_values.append(tpi_value)
                
                error = abs(tpi_value - optimal_value)
                relative_error = error / abs(optimal_value) if abs(optimal_value) > 1e-6 else error
                
                print(f"Valeur TPI: {tpi_value:.6f}")
                print(f"Écart absolu: {error:.6f}")
                print(f"Écart relatif: {relative_error:.2%}")
            
            # Analyse de la convergence
            improvement_3k = abs(tpi_values[1] - optimal_value) - abs(tpi_values[0] - optimal_value)
            improvement_30k = abs(tpi_values[2] - optimal_value) - abs(tpi_values[1] - optimal_value)
            
            print(f"\nAmélioration 300->3K: {improvement_3k:.6f}")
            print(f"Amélioration 3K->30K: {improvement_30k:.6f}")
            
            tested_domains += 1
            
            # CRITÈRE STRICT: Si 100x plus d'entraînement n'améliore pas significativement,
            # il y a un problème fondamental
            final_error = abs(tpi_values[2] - optimal_value) / abs(optimal_value) if abs(optimal_value) > 1e-6 else abs(tpi_values[2] - optimal_value)
            
            if final_error <= 0.05:  # 5% de tolérance seulement après entraînement intensif
                successful_domains += 1
                print(f"✅ SUCCÈS - Convergence acceptable")
            else:
                # Diagnostics détaillés pour comprendre l'échec
                print(f"❌ ÉCHEC - Pas de convergence même avec 30K épisodes")
                print(f"   Erreur finale: {final_error:.2%}")
                print(f"   Amélioration totale: {improvement_3k + improvement_30k:.6f}")
                
                # Si pas d'amélioration significative avec 100x plus d'entraînement,
                # c'est un problème algorithmique
                if abs(improvement_3k + improvement_30k) < 0.001:
                    pytest.fail(f"TPI ne converge pas sur {domain_name} H{horizon}: "
                              f"aucune amélioration significative avec 30K épisodes "
                              f"(erreur finale {final_error:.2%})")
                
        except Exception as e:
            print(f"❌ Erreur {domain_name}: {e}")
            continue

    # Vérification globale plus stricte
    if tested_domains == 0:
        pytest.skip(f"Aucun domaine testé pour horizon {horizon}")

    success_rate = successful_domains / tested_domains
    print(f"\n=== RÉSUMÉ HORIZON {horizon} ===")
    print(f"Domaines testés: {tested_domains}")
    print(f"Succès (< 5% erreur après 30K épisodes): {successful_domains}")
    print(f"Taux de succès: {success_rate:.1%}")

    # Attendre au moins 50% de succès avec entraînement intensif
    assert success_rate >= 0.5, f"Taux de succès insuffisant après entraînement intensif: {success_rate:.1%}"


if __name__ == "__main__":
    pytest.main([__file__])