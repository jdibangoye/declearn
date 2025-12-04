# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
"""
Test de validation : Politique optimale exhaustive vs TPI pour horizon T=1

Pour T=1, la solution optimale est simplement :
π* = argmax_{a₁,...,aₙ} E[R(s,a₁,...,aₙ) | b₀]

où b₀ est la croyance initiale sur les états.
"""

from declearn.envs.masplan import parse_dpomdp
from declearn.core.sequential_env import SequentialDecPOMDPSimulator
from declearn.core.types import make_sequential_dag
from declearn.core.tabular_tpi import TabularTPI
from declearn.core.sa_schedules import ThreeTimeScale
import numpy as np
import itertools

def exhaustive_search_horizon_1(spec):
    """
    Recherche exhaustive de la meilleure politique jointe pour horizon T=1.
    
    Calcule la VRAIE valeur V^π(s₀) en utilisant le modèle exact (T, R):
        V^π = Σ_s₀ P(s₀) * Σ_s' T(s'|s₀,π) * R(s₀,π,s')
    
    Returns:
        best_joint_action: tuple d'actions optimales
        best_value: valeur exacte de la meilleure politique
        all_values: dict {joint_action: value} pour toutes les combinaisons
    """
    spec.horizon = 1
    
    # Générer toutes les combinaisons d'actions jointes possibles
    action_spaces = [range(len(spec.actions[i])) for i in range(spec.n_agents)]
    all_joint_actions = list(itertools.product(*action_spaces))
    
    print("=" * 80)
    print("RECHERCHE EXHAUSTIVE - HORIZON T=1 (VALEUR EXACTE)")
    print("=" * 80)
    print(f"Nombre d'agents: {spec.n_agents}")
    print(f"États: {spec.states}")
    print(f"Actions agent 0: {spec.actions[0]}")
    print(f"Actions agent 1: {spec.actions[1]}")
    print(f"Combinaisons totales: {len(all_joint_actions)}")
    print(f"Méthode: Calcul exact avec modèle (T, R)")
    
    # Utiliser le modèle directement depuis spec
    # Distribution initiale: start_belief
    if spec.start_belief is not None:
        initial_belief = spec.start_belief
    elif spec.init_belief is not None:
        initial_belief = spec.init_belief
    else:
        # Si pas de croyance, état unique
        start_state = spec.start if isinstance(spec.start, str) else spec.states[spec.start]
        initial_belief = np.zeros(len(spec.states))
        initial_belief[spec.states.index(start_state)] = 1.0
    
    print(f"\nDistribution initiale:")
    for idx, prob in enumerate(initial_belief):
        if prob > 1e-10:
            print(f"  P({spec.states[idx]}) = {prob:.4f}")
    
    all_values = {}
    best_joint_action = None
    best_value = -np.inf
    
    for joint_action in all_joint_actions:
        # Convertir en noms d'actions pour affichage
        action_names = [spec.actions[i][joint_action[i]] for i in range(spec.n_agents)]
        
        print(f"\n  π={joint_action} = {tuple(action_names)}:")
        
        # À horizon T=1, pas de transition ! Seulement la récompense immédiate:
        # V^π = E_{s₀~b₀}[R(s₀, π)] = Σ_s₀ P(s₀) · R(s₀, π)
        value = 0.0
        
        for s0_idx, prob_s0 in enumerate(initial_belief):
            if prob_s0 < 1e-10:
                continue
            
            s0_name = spec.states[s0_idx]
            
            # Clé récompense: (state_idx, joint_action_tuple)
            rew_key = (s0_idx, joint_action)
            reward = spec.reward.get(rew_key, 0.0)
            
            if abs(reward) > 1e-6:
                print(f"    P({s0_name})={prob_s0:.4f} × R({s0_name},{tuple(action_names)})={reward:.2f} = {prob_s0 * reward:.6f}")
            else:
                print(f"    P({s0_name})={prob_s0:.4f} × R({s0_name},{tuple(action_names)})={reward:.2f} = 0.000000")
            
            value += prob_s0 * reward
        
        print(f"    ⟹ V^π = {value:.6f}")
        all_values[joint_action] = value
        
        if value > best_value:
            best_value = value
            best_joint_action = joint_action
        
        if len(all_joint_actions) <= 25:
            print(f"  π={joint_action}: V={value:.6f}")
    
    print("\n" + "=" * 80)
    print(f"✓ POLITIQUE OPTIMALE: {best_joint_action}")
    print(f"✓ VALEUR OPTIMALE (EXACTE): {best_value:.6f}")
    print("=" * 80)
    
    return best_joint_action, best_value, all_values


def evaluate_tpi_horizon_1(spec, episodes=1000, eval_episodes=1000):
    """
    Entraîne TPI sur horizon T=1 et évalue sa performance.
    """
    spec.horizon = 1
    
    dag = make_sequential_dag(spec.n_agents, 1)
    n_actions = {i: len(spec.actions[i]) for i in range(spec.n_agents)}
    
    # Taux d'apprentissage élevés pour convergence rapide sur T=1
    ts = ThreeTimeScale(alpha=0.5, beta=0.3, gamma=0.1)
    rng = np.random.default_rng(42)
    tpi = TabularTPI(dag, n_actions, ts, 1, rng)
    
    print("\n" + "=" * 80)
    print("ENTRAÎNEMENT TPI - HORIZON T=1")
    print("=" * 80)
    print(f"Épisodes d'entraînement: {episodes}")
    alpha_val = ts.alpha_base if hasattr(ts, 'alpha_base') else 0.5
    beta_val = ts.beta_base if hasattr(ts, 'beta_base') else 0.3
    gamma_val = ts.gamma_base if hasattr(ts, 'gamma_base') else 0.1
    print(f"Learning rates: α={alpha_val}, β={beta_val}, γ={gamma_val}")
    
    training_rewards = []
    
    # Phase d'entraînement
    for episode in range(episodes):
        sim = SequentialDecPOMDPSimulator(spec, seed=42 + episode, memory_m=1)
        type_tuple = sim.reset()
        
        episode_reward = 0.0
        done = False
        episode_samples = []
        
        # ε-greedy pour exploration
        epsilon = max(0.01, 1.0 - episode / (episodes * 0.5))
        
        step = 0
        while not done and step < spec.n_agents:
            agent_id = step % spec.n_agents
            k = episode * spec.n_agents + step
            
            theta = type_tuple
            
            # Politique ε-greedy
            if rng.random() < epsilon:
                action = rng.integers(0, n_actions[agent_id])
            else:
                action = tpi.act(agent_id, theta, greedy=True, k=k)
            
            next_type_tuple, reward, done, _ = sim.step(action)
            episode_reward += reward
            
            # Échantillonner a_next
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
        
        # Appliquer les mises à jour
        for i, (agent_id, sample_data) in enumerate(episode_samples):
            sample = {agent_id: sample_data}
            k_update = episode * spec.n_agents + i
            tpi.step(sample, k=k_update)
        
        if episode % (episodes // 10) == 0:
            avg_reward = np.mean(training_rewards[-100:]) if len(training_rewards) >= 100 else np.mean(training_rewards)
            q_entries = sum(len(q_dict) for q_dict in tpi.q_tables.values())
            print(f"  Episode {episode}/{episodes}: ε={epsilon:.3f}, R_avg={avg_reward:.4f}, Q_states={q_entries}")
    
    # Analyse des Q-tables et politiques
    print("\n  Analyse des Q-tables:")
    for u_pair in sorted(tpi.q_tables.keys()):
        q_dict = tpi.q_tables[u_pair]
        n_states = len(q_dict)
        all_q_values = [v for q_theta in q_dict.values() for v in q_theta if abs(v) > 1e-6]
        n_nonzero = len(all_q_values)
        
        if n_nonzero > 0:
            print(f"    u={u_pair}: {n_states} états, {n_nonzero} Q non-zéro, "
                  f"Q_range=[{min(all_q_values):.3f}, {max(all_q_values):.3f}]")
    
    print(f"\n  Nombre de politiques π_u créées: {len(tpi.policies)}")
    print(f"  Clés: {sorted(tpi.policies.keys())}")
    
    print("\n  Analyse des G-tables et politiques:")
    for u_pair in sorted(tpi.g_tables.keys()):
        agent_id = u_pair[0] if isinstance(u_pair, tuple) else u_pair
        g_dict = tpi.g_tables[u_pair]
        
        print(f"\n    u={u_pair} (agent {agent_id}): G-table avec {len(g_dict)} historiques")
        
        # Vérifier si la politique existe pour cette sous-paire
        if u_pair not in tpi.policies:
            print(f"      ⚠️  POLITIQUE π_{u_pair} MANQUANTE")
            for i, (h_key, g_values) in enumerate(list(g_dict.items())[:2]):
                print(f"        h={str(h_key)[:40]}: G={dict(g_values)}")
            continue
            
        policy_obj = tpi.policies[u_pair]
        
        if hasattr(policy_obj, 'logits'):
            print(f"      ✓ Politique π_{u_pair} existe: {len(policy_obj.logits)} clés dans logits")
            for i, (h_key, g_values) in enumerate(list(g_dict.items())[:2]):
                # MemoryPolicy utilise des clés (uidx, h)
                policy_key = (agent_id, h_key)
                if policy_key in policy_obj.logits:
                    policy_probs = policy_obj.probs(agent_id, h_key)
                    best_action = int(np.argmax(policy_probs))
                    action_name = spec.actions[agent_id][best_action]
                    probs_str = '[' + ', '.join(f'{p:.3f}' for p in policy_probs) + ']'
                    print(f"        h={str(h_key)[:40]}: G={dict(g_values)}")
                    print(f"          → π={probs_str} → a={best_action}({action_name})")
                else:
                    print(f"        h={str(h_key)[:40]}: G={dict(g_values)}, ⚠️ PAS DANS π.logits")
        else:
            print(f"      ⚠️ Politique sans logits")
    
    # Phase d'évaluation
    print(f"\n  Évaluation sur {eval_episodes} épisodes (greedy)...")
    
    # DEBUG: Afficher les premiers épisodes
    print(f"\n  DEBUG: Premiers 3 épisodes d'évaluation:")
    
    eval_rewards = []
    policy_counts = {}  # Pour compter les politiques jointes choisies
    
    for episode in range(eval_episodes):
        sim = SequentialDecPOMDPSimulator(spec, seed=episode, memory_m=1)
        type_tuple = sim.reset()
        
        episode_reward = 0.0
        done = False
        step = 0
        joint_policy = []
        
        if episode < 3:
            print(f"\n    Épisode {episode}:")
            print(f"      État initial: {type_tuple.state}")
            print(f"      type_tuple.t: {type_tuple.t if hasattr(type_tuple, 't') else 'N/A'}")
            if hasattr(type_tuple, 'joint_histories'):
                print(f"      joint_histories:")
                for i, hist in enumerate(type_tuple.joint_histories):
                    print(f"        Agent {i}: {hist}")
        
        while not done and step < spec.n_agents:
            agent_id = step % spec.n_agents
            k = episode * spec.n_agents + step
            
            # Extraire l'historique vu par l'agent
            u = (agent_id, type_tuple.t if hasattr(type_tuple, 't') else 0)
            h = tpi._extract_private_history(agent_id, type_tuple)
            
            action = tpi.act(agent_id, type_tuple, greedy=True, k=k)
            joint_policy.append(action)
            
            if episode < 3:
                action_name = spec.actions[agent_id][action]
                print(f"      Agent {agent_id}: u={u}, h={h} → a={action}({action_name})")
            
            next_type_tuple, reward, done, _ = sim.step(action)
            episode_reward += reward
            
            type_tuple = next_type_tuple
            step += 1
        
        # Enregistrer la politique jointe
        if len(joint_policy) == spec.n_agents:
            joint_policy_tuple = tuple(joint_policy)
            policy_counts[joint_policy_tuple] = policy_counts.get(joint_policy_tuple, 0) + 1
        
        eval_rewards.append(episode_reward)
    
    # Afficher les politiques apprises
    print(f"\n  Politiques jointes choisies par TPI (greedy):")
    for joint_policy, count in sorted(policy_counts.items(), key=lambda x: -x[1]):
        action_names = [spec.actions[i][joint_policy[i]] for i in range(spec.n_agents)]
        freq = count / eval_episodes
        print(f"    π={joint_policy} = {tuple(action_names)}: {count}/{eval_episodes} fois ({freq:.1%})")
    
    tpi_value = np.mean(eval_rewards)
    tpi_std = np.std(eval_rewards)
    
    print(f"\n✓ VALEUR TPI: {tpi_value:.6f} ± {tpi_std:.6f}")
    print("=" * 80)
    
    return tpi_value, tpi_std


def main():
    path = 'declearn/envs/masplan_specs/BroadcastChannel.dpomdp'
    spec = parse_dpomdp(path)
    
    print("\n" + "=" * 80)
    print("TEST DE VALIDATION: EXHAUSTIF vs TPI (HORIZON T=1)")
    print("=" * 80)
    print(f"Domaine: BroadcastChannel")
    print(f"Nombre d'agents: {spec.n_agents}")
    print(f"Horizon: T=1")
    
    # 1. Recherche exhaustive (valeur exacte)
    optimal_policy, optimal_value, all_values = exhaustive_search_horizon_1(spec)
    
    # 2. TPI avec plus d'échantillons
    tpi_value, tpi_std = evaluate_tpi_horizon_1(spec, episodes=10000, eval_episodes=5000)
    
    # 3. Comparaison
    print("\n" + "=" * 80)
    print("COMPARAISON FINALE")
    print("=" * 80)
    print(f"Politique optimale (exhaustive): {optimal_policy}")
    print(f"  Valeur optimale: {optimal_value:.6f}")
    print(f"\nPolitique TPI:")
    print(f"  Valeur TPI: {tpi_value:.6f} ± {tpi_std:.6f}")
    print(f"\nÉcart:")
    print(f"  Absolu: {abs(tpi_value - optimal_value):.6f}")
    if abs(optimal_value) > 1e-6:
        relative_error = abs(tpi_value - optimal_value) / abs(optimal_value)
        print(f"  Relatif: {relative_error:.2%}")
        
        if relative_error < 0.05:
            print(f"\n✅ SUCCÈS: TPI atteint l'optimal (écart < 5%)")
        elif relative_error < 0.20:
            print(f"\n⚠️  ACCEPTABLE: TPI proche de l'optimal (écart < 20%)")
        else:
            print(f"\n❌ ÉCHEC: TPI loin de l'optimal (écart > 20%)")
            print(f"\n💡 Le problème fondamental de TPI n'est pas résolu.")
    else:
        print(f"\n⚠️  Valeur optimale proche de zéro, écart relatif non significatif")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
