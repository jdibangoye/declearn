# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from typing import Dict, Tuple, Any, List, Optional
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from .sequential_env import TypeTuple
from .memory_policy import MemoryPolicy
from .sa_schedules import ThreeTimeScale  # Import des schedules théoriques


class TabularTPI:
    """
    Tabular Type-based Policy Iteration pour Dec-POMDP
    Implémentation correcte avec sous-paires u=(i,t) et schedules théoriques
    """
    
    def __init__(self, dag, n_actions_dict, timescale, memory_m, rng, exploration_schedule=None):
        """Initialize TPI pour tous les agents du Dec-POMDP"""
        self.dag = dag
        self.n_actions_dict = n_actions_dict
        self.timescale = timescale  # ThreeTimeScale object
        self.memory_m = memory_m
        self.rng = rng
        
        # Schedule d'exploration GLIE pour TPI théorique
        if exploration_schedule is None:
            from .sa_schedules import ExplorationSchedule
            exploration_schedule = ExplorationSchedule()
        self.exploration_schedule = exploration_schedule
        
        # CORRECTION TPI THÉORIQUE : Toutes les structures indexées par sous-paires u=(i,t)
        # Structure : q_tables[u][θ] = Q_u(θ) où u=(i,t)
        # Structure : g_tables[u][h_i][a_i] = G_u(h_i, a_i) où u=(i,t)
        # Structure : policies[u] = π_u où u=(i,t)
        self.q_tables: Dict[Tuple[int, int], Dict] = {}  # Par sous-paire u=(i,t)
        self.g_tables: Dict[Tuple[int, int], Dict[Tuple, Dict[int, float]]] = {}
        
        # Politiques π_u indexées par u=(i,t) - créées dynamiquement selon besoin
        self.policies: Dict[Tuple[int, int], MemoryPolicy] = {}
        
        # Compteurs pour les échelles de temps
        self.step_count = 0
        
        # Construire le mapping successeur depuis le DAG existant
        self._build_successor_map()
        
    def _build_successor_map(self):
        """Créer mapping (agent, time) → successor depuis le DAG existant"""
        
        # Créer un mapping (agent, time) → successor (agent, time)  
        self.successor_map = {}
        
        # SubStageDAG avec ordre topologique correct
        for stage in self.dag.stages:
            if stage.successor is not None:
                succ_stage = self.dag.stages[stage.successor]
                # ATTENTION: DAG utilise agent 1-indexé, TabularTPI utilise 0-indexé
                self.successor_map[(stage.agent-1, stage.time)] = (succ_stage.agent-1, succ_stage.time)
            else:
                self.successor_map[(stage.agent-1, stage.time)] = None  # Terminal

    def act(self, uidx: int, theta: TypeTuple, greedy: bool = False, k: int = 0, use_q_values: bool = False) -> int:
        """Action de l'agent uidx basée sur le type complet theta avec exploration GLIE.
        
        Args:
            uidx: Agent index
            theta: Type tuple complet
            greedy: Si True, utilise politique greedy, sinon échantillonne
            k: Compteur pour exploration schedule
            use_q_values: Si True, utilise Q-values directement au lieu de G-tables (bypass G)
        """
        
        # Extraire t du theta pour former u = (uidx, t)
        if hasattr(theta, 't'):
            t = theta.t
        elif isinstance(theta, tuple) and len(theta) >= 2:
            t = theta[1]  # Assume (state, t) format
        else:
            t = 0  # Fallback
            
        u = (uidx, t)
        
        # GLIE behaviour: ς_u(·|h) = (1-ε_k)σ_u(·|h) + ε_k*Unif(Acts_u)
        if not greedy:
            epsilon_k = self.exploration_schedule.epsilon(k)
            if self.rng.random() < epsilon_k:
                # Exploration uniforme
                return self.rng.choice(self.n_actions_dict[uidx])
        
        # OPTION 1: Utiliser Q-values directement (bypass G-tables)
        if use_q_values and u in self.q_tables and theta in self.q_tables[u]:
            q_vals = self.q_tables[u][theta]
            if isinstance(q_vals, np.ndarray):
                return int(np.argmax(q_vals))
            else:
                # Dict format
                return max(q_vals.items(), key=lambda x: x[1])[0]
        
        # OPTION 2: Utiliser G-tables via policy (comportement original)
        # Créer la politique π_u si elle n'existe pas encore
        if u not in self.policies:
            self.policies[u] = MemoryPolicy(self.n_actions_dict[uidx])

        pol = self.policies[u]

        # Extraction de l'historique privé h_i avec limitation de mémoire
        h = self._extract_private_history(uidx, theta)

        if not greedy:
            return pol.sample(uidx, h, self.rng)
        else:
            return pol.greedy(uidx, h)

    def step(self, sample: Dict[int, Dict], k: int):
        """
        Mise à jour TPI avec échantillon multi-agent
        
        Args:
            sample: {agent_id: {"theta": ..., "a": ..., "r": ..., ...}}
            k: Numéro d'étape global
        """
        self.step_count = k
        
        # Mise à jour pour chaque agent dans l'échantillon
        for uidx, agent_sample in sample.items():
            # Les politiques sont maintenant créées dynamiquement dans act()
            # Plus besoin de vérification préalable ici
            self._update_agent(uidx, agent_sample, k)
    
    def trial(self, sim, u_idx: int, theta, k: int, epsilon_k: float):
        """
        Procédure Trial récursive selon l'algorithme théorique (variante arrière).
        
        Ligne 6-18 de l'algorithme :
        - Si u = u_T : retour
        - Sinon : échantillonne action, fait step, appel récursif, puis mise à jour
        
        Args:
            sim: Simulateur pour générer (θ',r) ~ gen(·|θ,a)
            u_idx: Indice du nœud courant dans le DAG
            theta: Type courant θ
            k: Compteur global pour schedules
            epsilon_k: Paramètre d'exploration GLIE
            
        Returns:
            Récompense totale de l'épisode
        """
        # Ligne 6 : if u = u_T then return
        stage = self.dag.stages[u_idx]
        if stage.successor is None:  # Terminal
            return 0.0
        
        uidx = stage.agent
        u_pair = (uidx, stage.time)
        
        # Convertir agent DAG (1-based) vers index actions (0-based)
        agent_idx = uidx - 1 if uidx > 0 else 0
        
        # Créer politique si nécessaire
        if u_pair not in self.policies:
            self.policies[u_pair] = MemoryPolicy(self.n_actions_dict[agent_idx])
        
        # Ligne 7-8 : définir ς_u et échantillonner a
        h = self._extract_private_history(agent_idx, theta)
        
        # Politique de comportement : ς_u = (1-ε_k)σ_u + ε_k·Unif
        if self.rng.random() < epsilon_k:
            a = self.rng.integers(0, self.n_actions_dict[agent_idx])
        else:
            a = self.policies[u_pair].sample(agent_idx, h, self.rng)
        
        # Ligne 9 : (θ',r) ~ gen(·|θ,a), u' ← Succ(u)
        theta_next, r, done, _ = sim.step(a)
        total_reward = r
        
        u_next_idx = stage.successor
        
        # Ligne 10 : Trial(u',θ') - appel récursif
        if not done and u_next_idx is not None:
            total_reward += self.trial(sim, u_next_idx, theta_next, k+1, epsilon_k)
        
        # Ligne 11 : h' ← Priv_u'(θ'), sample a' ~ σ_u'(·|h')
        # IMPORTANT : a' échantillonné de σ (politique cible) pas ς
        if u_next_idx is not None:
            next_stage = self.dag.stages[u_next_idx]
            u_next_pair = (next_stage.agent, next_stage.time)
            next_agent_idx = next_stage.agent - 1 if next_stage.agent > 0 else 0
            
            if not done:
                if u_next_pair not in self.policies:
                    self.policies[u_next_pair] = MemoryPolicy(self.n_actions_dict[next_agent_idx])
                
                h_next = self._extract_private_history(next_agent_idx, theta_next)
                a_next = self.policies[u_next_pair].sample(next_agent_idx, h_next, self.rng)
            else:
                # Terminal : a_next n'a pas d'importance pour le bootstrap
                a_next = 0
        else:
            # Pas de successeur (ne devrait pas arriver car on vérifie terminal au début)
            u_next_pair = u_pair
            a_next = 0
        
        # Ligne 12-17 : Mises à jour Q, G, politique
        # Ces mises à jour se font TOUJOURS, même si terminal
        self._update_q_table_trial(u_pair, theta, a, r, u_next_pair, theta_next, a_next, k)
        self._update_g_table_trial(u_pair, agent_idx, theta, a, h, k)
        self._update_policy_trial(u_pair, agent_idx, h, k)
        
        return total_reward

    def _update_agent(self, uidx: int, sample: Dict, k: int):
        """Mise à jour TPI complète pour agent uidx"""
        
        # 1. Q-table update (échelle rapide)
        theta = sample["theta"]
        a = sample["a"]
        r = sample["r"] 
        theta_next = sample["theta_next"]
        a_next = sample.get("a_next", None)  # Action échantillonnée à θ_next
        
        self._update_q_table(uidx, theta, a, r, theta_next, a_next, k)
        
        # 2. G-table update pour toutes sous-paires (échelle intermédiaire)
        self._update_g_table(uidx, sample, k)
        
        # 3. Policy update basée sur G_{(uidx,t)} (échelle lente)
        self._update_policy(uidx, sample, k)

    def _update_q_table(self, uidx: int, theta: Tuple, a: int, r: float, theta_next: Tuple, a_next: Optional[int], k: int):
        """Mise à jour Q-table avec ThreeTimeScale - Q_u(θ) où u=(i,t)
        
        Implémente SARSA : Δq_u(θ,a) ← α_k[r + q_{u'}(θ',a') - q_u(θ,a)]
        où a' est l'action échantillonnée à θ_next selon la politique.
        """
        
        # Utiliser le schedule théorique pour Q-learning (échelle rapide)
        current_timescale = self.timescale.decay(k)
        alpha_k = current_timescale.alpha
        
        # Extraire t du theta et theta_next
        if hasattr(theta, 't'):
            t = theta.t
        elif isinstance(theta, tuple) and len(theta) >= 2:
            t = theta[1]  # Assume (state, t) format
        else:
            t = 0  # Fallback
            
        if hasattr(theta_next, 't'):
            t_next = theta_next.t
        elif isinstance(theta_next, tuple) and len(theta_next) >= 2:
            t_next = theta_next[1]  # Assume (state, t) format
        else:
            t_next = 0  # Fallback
        
        # Sous-paire actuelle u = (agent_id, t)
        u_pair = (uidx, t)
        
        # CORRECTION TPI: u_next = successeur selon le DAG (pas nécessairement même agent !)
        u_next_pair = self.successor_map.get(u_pair, u_pair)  # Fallback si pas trouvé
        
        # DIAGNOSTIC: Traquer les transitions inter-sous-pas pour comprendre le ruissellement
        if hasattr(self, '_transition_count'):
            self._transition_count += 1
        else:
            self._transition_count = 1
            self._inter_subpair_count = 0
            self._same_subpair_count = 0
            
        if u_pair != u_next_pair:
            self._inter_subpair_count += 1
            if self._transition_count <= 10:  # Premiers exemples
                print(f"    TRANSITION INTER-SOUS-PAS: {u_pair} → {u_next_pair}, r={r:.3f}")
        else:
            self._same_subpair_count += 1
        
        # Utiliser directement les objets comme clés hashables
        theta_key = theta
        theta_next_key = theta_next
    
        # Initialisation de la Q-table pour cette sous-paire si nécessaire
        if u_pair not in self.q_tables:
            self.q_tables[u_pair] = {}
            
        if theta_key not in self.q_tables[u_pair]:
            self.q_tables[u_pair][theta_key] = np.zeros(self.n_actions_dict[uidx])
        if theta_next_key not in self.q_tables[u_pair]:
            self.q_tables[u_pair][theta_next_key] = np.zeros(self.n_actions_dict[uidx])
        
        # CORRECTION: TPI utilise la sous-paire SUCCESSEUR pour q_{u'}(θ',a')
        current_q = self.q_tables[u_pair][theta_key][a]
        
        # Initialiser la table de la sous-paire suivante si nécessaire
        if u_next_pair not in self.q_tables:
            self.q_tables[u_next_pair] = {}
        if theta_next_key not in self.q_tables[u_next_pair]:
            # CORRECTION: utiliser les actions du bon agent pour u_next_pair
            next_agent = u_next_pair[0] if u_next_pair else uidx
            self.q_tables[u_next_pair][theta_next_key] = np.zeros(self.n_actions_dict[next_agent])
            
        # TPI utilise SARSA : q_{u'}(θ', a') où a' est échantillonné de la politique
        if a_next is not None:
            # SARSA : utiliser l'action effectivement prise à θ_next
            next_q_value = self.q_tables[u_next_pair][theta_next_key][a_next]
        else:
            # Fallback si a_next non fourni (ne devrait pas arriver en production)
            # Pour la rétrocompatibilité avec les anciens tests
            next_q_value = np.max(self.q_tables[u_next_pair][theta_next_key])
        
        td_error = r + 1.0 * next_q_value - current_q  # Pas de discount pour TPI
        new_q = current_q + alpha_k * td_error
        self.q_tables[u_pair][theta_key][a] = new_q
        
        # DEBUG désactivé pour performance
        # if hasattr(self, '_q_update_count'):
        #     self._q_update_count += 1

    def _update_g_table(self, uidx: int, sample: Dict, k: int):
        """Mise à jour G-tables avec ThreeTimeScale - pour agent uidx seulement"""
    
        # Utiliser le schedule théorique pour G-tables (échelle intermédiaire)
        current_timescale = self.timescale.decay(k)
        beta_k = current_timescale.beta

        theta = sample["theta"]
        theta_key = theta  # Utiliser directement le tuple comme clé
        
        # Extraire t du theta (même logique que _update_q_table)
        if hasattr(theta, 't'):
            t = theta.t
        elif isinstance(theta, tuple) and len(theta) >= 2:
            t = theta[1]  # Assume (state, t) format
        else:
            t = 0  # Fallback
        
        # Sous-paire u = (uidx, t) pour l'agent courant
        u = (uidx, t)
        
        # Extraire h_i et a_i pour l'agent uidx depuis le sample
        h_i = self._extract_private_history_from_sample(uidx, sample)
        a_i = self._extract_private_action_from_sample(uidx, sample)
        
        if h_i is None or a_i is None:
            return
        
        # Vérifier que les Q-tables existent
        if u not in self.q_tables or theta_key not in self.q_tables[u]:
            return
        
        # Correspondance avec Q_u(θ, a_i) pour u = (uidx, t)
        q_values_i = self.q_tables[u][theta_key]
        
        if a_i < len(q_values_i):
            q_value_for_ai = q_values_i[a_i]
            
            # Initialiser G_u(h_i, a_i) si nécessaire
            if u not in self.g_tables:
                self.g_tables[u] = {}
            if h_i not in self.g_tables[u]:
                self.g_tables[u][h_i] = {}
            if a_i not in self.g_tables[u][h_i]:
                self.g_tables[u][h_i][a_i] = 0.0
            
            # CORRECTION: Mise à jour G à chaque fois (pas seulement à l'initialisation)
            current_g = self.g_tables[u][h_i][a_i]
            g_error = q_value_for_ai - current_g
            new_g = current_g + beta_k * g_error
            self.g_tables[u][h_i][a_i] = new_g
            
        # DEBUG désactivé pour performance  
        # if hasattr(self, '_g_update_count'):
        #     self._g_update_count += 1

    def _update_policy(self, uidx: int, sample: Dict, k: int):
        """Mise à jour politique avec ThreeTimeScale"""
    
        # Utiliser le schedule théorique pour policies (échelle lente)
        current_timescale = self.timescale.decay(k)
        gamma_k = current_timescale.gamma  # CORRECTION : gamma = slowest
    
        theta = sample["theta"]
        t = theta.t if hasattr(theta, 't') else sample.get("t", 0)
        
        # Sous-paire u = (uidx, t) pour cette politique spécifique
        u = (uidx, t)
        
        # Historique privé de l'agent uidx
        h_i = self._extract_private_history_from_sample(uidx, sample)
        
        # DIAGNOSTIC: Pourquoi seulement 15/32 mises à jour ?
        if hasattr(self, '_policy_debug_count'):
            self._policy_debug_count += 1
        else:
            self._policy_debug_count = 1
        
        if h_i is None:
            if hasattr(self, '_policy_skip_none_h'):
                self._policy_skip_none_h += 1
            else:
                self._policy_skip_none_h = 1
            return
        
        if u not in self.g_tables:
            if hasattr(self, '_policy_skip_no_g'):
                self._policy_skip_no_g += 1
            else:
                self._policy_skip_no_g = 1
            return
        
        # Créer la politique π_u si elle n'existe pas encore
        if u not in self.policies:
            self.policies[u] = MemoryPolicy(self.n_actions_dict[uidx])
            
        # Récupérer G_u(h_i, a_i) pour toutes les actions a_i de l'agent uidx
        if h_i in self.g_tables[u]:
            g_values_dict = self.g_tables[u][h_i]
            
            # Convertir en array pour mirror_update
            g_values = np.zeros(self.n_actions_dict[uidx])
            for action, g_val in g_values_dict.items():
                if action < len(g_values):
                    g_values[action] = g_val
            
            # Mise à jour avec schedule théorique γ_k
            self.policies[u].mirror_update(uidx, h_i, g_values, gamma_k)
            
            if hasattr(self, '_policy_updates_success'):
                self._policy_updates_success += 1
            else:
                self._policy_updates_success = 1
        else:
            if hasattr(self, '_policy_skip_no_h_in_g'):
                self._policy_skip_no_h_in_g += 1
            else:
                self._policy_skip_no_h_in_g = 1

    def _extract_private_history(self, uidx: int, theta: TypeTuple) -> Tuple:
        """Extrait l'historique privé action-observation h_i de l'agent i depuis TypeTuple
        
        L'historique suit l'ordre temporel : h = (a₀, o₁, a₁, o₂, ...)
        où l'action aᵢ précède toujours l'observation oᵢ₊₁
        """
        
        if uidx >= len(theta.joint_histories):
            return ()
        
        agent_history = theta.joint_histories[uidx]
        
        # Construire historique action-observation pour TPI stochastique
        if 'obs' in agent_history and 'acts' in agent_history:
            obs_list = agent_history['obs']
            acts_list = agent_history['acts']
            
            # Filtrer les actions et observations invalides (-1 = dummy initial)
            valid_acts = [a for a in acts_list if a != -1]
            valid_obs = [o for o in obs_list if o != -1]
            
            # Construire paires (action, observation) dans l'ordre temporel
            # L'action i correspond à l'observation i+1
            ao_pairs = []
            pair_count = min(len(valid_acts), len(valid_obs))
            
            for i in range(pair_count):
                ao_pairs.append(('ao', valid_acts[i], valid_obs[i]))
            
            return tuple(ao_pairs[-self.memory_m:])
        
        # Fallback : observations seules (filtrer -1)
        elif 'obs' in agent_history:
            valid_obs = [o for o in agent_history['obs'] if o != -1]
            return tuple(valid_obs[-self.memory_m:])
        
        return ()

    def _extract_private_history_from_sample(self, agent_i: int, sample: Dict) -> Tuple:
        """Extrait l'historique privé action-observation h_i de l'agent i depuis le sample
        
        Pour TPI avec politique stochastique, l'historique doit inclure les actions:
        h_i = (a_0, o_1, a_1, o_2, ...) jusqu'à la mémoire m
        """
        
        # Méthode 1 : depuis theta dans le sample
        if "theta" in sample:
            theta = sample["theta"]
            if hasattr(theta, 'joint_histories') and agent_i < len(theta.joint_histories):
                agent_history = theta.joint_histories[agent_i]
                
                # Construire historique action-observation pour TPI stochastique
                if 'obs' in agent_history and 'acts' in agent_history:
                    obs_list = agent_history['obs']
                    acts_list = agent_history['acts']
                    
                    # IMPORTANT: L'historique h est ce qui est observé AVANT de choisir l'action a
                    # L'action courante sample["a"] ne fait PAS partie de h !
                    # h est l'historique utilisé pour choisir a via π(a|h)
                    
                    # Filtrer les actions et observations invalides (-1 = dummy initial)
                    valid_acts = [a for a in acts_list if a != -1]
                    valid_obs = [o for o in obs_list if o != -1]
                    
                    # Construire paires (action, observation) dans l'ordre temporel
                    # L'action i correspond à l'observation i+1
                    ao_pairs = []
                    pair_count = min(len(valid_acts), len(valid_obs))
                    
                    for i in range(pair_count):
                        ao_pairs.append(('ao', valid_acts[i], valid_obs[i]))
                    
                    # Debug désactivé pour performance
                    # if hasattr(self, '_debug_history_once') and not getattr(self, '_debug_history_once', True):
                    #     pass
                    # else:
                    #     print(f"    HISTORIQUE AO Agent {agent_i}: acts={complete_acts}, obs={obs_list} -> paires={ao_pairs}")
                    #     setattr(self, '_debug_history_once', True)
                    
                    # Limiter à la mémoire m et retourner tuple hashable
                    return tuple(ao_pairs[-self.memory_m:])
                
                # Fallback : observations seules (filtrer -1)
                elif 'obs' in agent_history:
                    valid_obs = [o for o in agent_history['obs'] if o != -1]
                    return tuple([('o', obs) for obs in valid_obs[-self.memory_m:]])
        
        # Méthode 2 : depuis clé h spécifique à l'agent (si sample mono-agent)
        if "h" in sample and "u" in sample:
            u_sample = sample["u"]
            # u peut être soit agent_i soit (agent_i, t)
            if u_sample == agent_i or (isinstance(u_sample, tuple) and len(u_sample) == 2 and u_sample[0] == agent_i):
                h = sample["h"]
                return h if isinstance(h, tuple) else tuple(h) if hasattr(h, '__iter__') else (h,)
            
        return None

    def _extract_private_action_from_sample(self, agent_i: int, sample: Dict) -> int:
        """Extrait l'action privée a_i de l'agent i depuis le sample"""
        
        # Méthode 1 : action spécifique à l'agent (si échantillon joint)
        a_key = f"a_{agent_i}"
        if a_key in sample:
            return sample[a_key]
        
        # Méthode 2 : action globale (si sample mono-agent pour cet agent)
        if "a" in sample and "u" in sample:
            u_sample = sample["u"]
            # u peut être soit agent_i soit (agent_i, t)
            if u_sample == agent_i or (isinstance(u_sample, tuple) and len(u_sample) == 2 and u_sample[0] == agent_i):
                return sample["a"]
        
        return None

    def get_g_table_stats(self):
        """Diagnostics des G-tables par sous-paires"""
        stats = {}
        for u, g_table in self.g_tables.items():
            agent_i, t = u
            total_entries = sum(len(actions) for actions in g_table.values())
            stats[f"G_{agent_i},{t}"] = {
                'histories': len(g_table),
                'total_entries': total_entries
            }
        return stats

    def get_learning_rate_diagnostics(self, k: int) -> Dict:
        """Diagnostics des pas d'apprentissage à l'étape k"""
        current_timescale = self.timescale.decay(k)
        
        return {
            'step': k,
            'alpha_k': current_timescale.alpha,   # CORRECTION
            'beta_k': current_timescale.beta,     # CORRECTION
            'gamma_k': current_timescale.gamma,   # CORRECTION
            'ratio_alpha_beta': current_timescale.alpha / current_timescale.beta if current_timescale.beta > 0 else float('inf'),
            'ratio_beta_gamma': current_timescale.beta / current_timescale.gamma if current_timescale.gamma > 0 else float('inf'),
            'scales_respected': current_timescale.alpha > current_timescale.beta > current_timescale.gamma > 0
        }
    
    def analyze_tables_density(self, k: int):
        """Analyser la densité des tables Q et G"""
        # Compter tous les éléments non-zéro dans toutes les sous-tables Q
        q_nonzero = sum(1 for q_table_u in self.q_tables.values() 
                       for q_theta in q_table_u.values()
                       for v in q_theta if abs(v) > 1e-6)
        q_total = sum(len(q_theta) for q_table_u in self.q_tables.values() 
                     for q_theta in q_table_u.values())
        
        # Compter tous les éléments non-zéro dans toutes les sous-tables G  
        g_nonzero = sum(1 for g_table_u in self.g_tables.values()
                       for g_h in g_table_u.values()  
                       for v in g_h.values() if abs(v) > 1e-6)
        g_total = sum(len(g_h) for g_table_u in self.g_tables.values()
                     for g_h in g_table_u.values())
        
        print(f"\nTABLE DENSITY @ k={k}:")
        print(f"  Q-table: {q_nonzero}/{q_total} non-zéro ({100*q_nonzero/q_total:.1f}%)")
        print(f"  G-table: {g_nonzero}/{g_total} non-zéro ({100*g_nonzero/g_total:.1f}%)")
        
        if q_nonzero > 0:
            q_values = [v for q_table_u in self.q_tables.values() 
                       for q_theta in q_table_u.values()
                       for v in q_theta if abs(v) > 1e-6]
            print(f"  Q-values: min={min(q_values):.4f}, max={max(q_values):.4f}, avg={sum(q_values)/len(q_values):.4f}")
        
        if g_nonzero > 0:
            g_values = [v for g_table_u in self.g_tables.values()
                       for g_h in g_table_u.values()  
                       for v in g_h.values() if abs(v) > 1e-6]
            print(f"  G-values: min={min(g_values):.4f}, max={max(g_values):.4f}, avg={sum(g_values)/len(g_values):.4f}")
            
        return {"q_density": q_nonzero/q_total, "g_density": g_nonzero/g_total}
    
    # =========================================================================
    # Méthodes pour la variante Trial (propagation arrière récursive)
    # =========================================================================
    
    def _update_q_table_trial(self, u_pair: Tuple[int, int], theta, a: int, r: float,
                               u_next_pair: Tuple[int, int], theta_next, a_next: int, k: int):
        """
        Ligne 12 de l'algorithme Trial :
        Δq_u(θ,a) ← α[r + q_u'(θ',a') - q_u(θ,a)]
        
        Note: Pas de decay temporel (α, β, γ constants dans cette variante)
        """
        # Constantes (pas de decay dans la variante simplifiée)
        alpha = self.timescale.alpha_base if hasattr(self.timescale, 'alpha_base') else 0.1
        
        # Convertir DAG agent (1-based) vers index actions (0-based)
        agent_idx = u_pair[0] - 1 if u_pair[0] > 0 else 0
        next_agent_idx = u_next_pair[0] - 1 if u_next_pair[0] > 0 else 0
        
        # Initialiser les tables
        if u_pair not in self.q_tables:
            self.q_tables[u_pair] = {}
        if theta not in self.q_tables[u_pair]:
            self.q_tables[u_pair][theta] = np.zeros(self.n_actions_dict[agent_idx])
        
        if u_next_pair not in self.q_tables:
            self.q_tables[u_next_pair] = {}
        if theta_next not in self.q_tables[u_next_pair]:
            self.q_tables[u_next_pair][theta_next] = np.zeros(self.n_actions_dict[next_agent_idx])
        
        # Ligne 12 : Δq_u(θ,a) ← α[r + q_u'(θ',a') - q_u(θ,a)]
        current_q = self.q_tables[u_pair][theta][a]
        next_q = self.q_tables[u_next_pair][theta_next][a_next]
        
        delta_q = alpha * (r + next_q - current_q)
        
        # Ligne 14 : q_u(θ,a) ← q_u(θ,a) + Δq_u(θ,a)
        self.q_tables[u_pair][theta][a] = current_q + delta_q
    
    def _update_g_table_trial(self, u_pair: Tuple[int, int], uidx: int, theta, a: int, h, k: int):
        """
        Ligne 13 de l'algorithme Trial :
        Δg_u(h,a) ← β[q_u(θ,a) - g_u(h,a)]
        """
        beta = self.timescale.beta_base if hasattr(self.timescale, 'beta_base') else 0.05
        
        # Initialiser G-table
        if u_pair not in self.g_tables:
            self.g_tables[u_pair] = {}
        if h not in self.g_tables[u_pair]:
            self.g_tables[u_pair][h] = np.zeros(self.n_actions_dict[uidx])
        
        # Ligne 13 : Δg_u(h,a) ← β[q_u(θ,a) - g_u(h,a)]
        if u_pair in self.q_tables and theta in self.q_tables[u_pair]:
            q_value = self.q_tables[u_pair][theta][a]
            current_g = self.g_tables[u_pair][h][a]
            
            delta_g = beta * (q_value - current_g)
            
            # Ligne 15 : g_u(h,a) ← g_u(h,a) + Δg_u(h,a)
            self.g_tables[u_pair][h][a] = current_g + delta_g
    
    def _update_policy_trial(self, u_pair: Tuple[int, int], agent_idx: int, h, k: int):
        """
        Ligne 16-17 de l'algorithme Trial :
        μ_u(·|h) ∈ argmax_b g_u(h,b)
        σ_u(·|h) ← (1-γ)σ_u(·|h) + γ·μ_u(·|h)
        """
        gamma = self.timescale.gamma_base if hasattr(self.timescale, 'gamma_base') else 0.01
        
        if u_pair not in self.g_tables or h not in self.g_tables[u_pair]:
            return
        
        # Créer politique si nécessaire
        if u_pair not in self.policies:
            self.policies[u_pair] = MemoryPolicy(self.n_actions_dict[agent_idx])
        
        # Ligne 16-17 : μ_u(·|h) ∈ argmax_b g_u(h,b), σ_u ← (1-γ)σ_u + γ·μ_u
        g_values = self.g_tables[u_pair][h]  # Déjà un numpy array
        
        # Mise à jour par mirror descent : même mécanisme que _update_policy
        self.policies[u_pair].mirror_update(agent_idx, h, g_values, gamma)