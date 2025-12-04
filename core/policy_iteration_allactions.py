# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.

"""
TPI-AllActions: Type-based Policy Iteration avec backup de toutes les actions

Approche théoriquement robuste :
- Évaluation on-policy pure (pas de GLIE)
- Backup de TOUTES les actions a ∈ A (pas seulement celle prise)
- Successeur en espérance V' = Σ_b σ(b|h') Q(θ',b)
- Amélioration basée sur G-tables : σ(h) ← argmax_a G(h,a)
"""

from typing import Dict, Tuple, Any, List, Optional
from collections import defaultdict
import numpy as np

from .sequential_env import TypeTuple
from .memory_policy import MemoryPolicy


class PolicyIterationAllActions:
    """
    Policy Iteration avec backup de toutes les actions
    
    Algorithme :
        RÉPÉTER jusqu'à convergence :
            1. ÉVALUATION : Générer E épisodes avec σ courante (on-policy pur)
                           Pour chaque substage :
                               - Action on-policy : a_on ~ σ(·|h)
                               - Backup TOUTES actions a ∈ A
                               - V' = Σ_b σ(b|h') Q(θ',b)
            
            2. AMÉLIORATION : σ(h) ← argmax_a G(h,a)
    """
    
    def __init__(
        self,
        dag,
        n_actions_dict: Dict[int, int],
        n_iterations: int = 20,
        n_eval_episodes: int = 10000,
        alpha: float = 0.1,
        beta: float = 0.01,
        memory_m: int = 1,
        rng=None
    ):
        """
        Args:
            dag: DAG des substages
            n_actions_dict: {agent_id: n_actions}
            n_iterations: Nombre d'itérations de Policy Iteration
            n_eval_episodes: Nombre d'épisodes d'évaluation par itération
            alpha: Learning rate pour Q-tables (rapide)
            beta: Learning rate pour G-tables (lent, β << α)
            memory_m: Taille mémoire pour historiques
            rng: Random number generator
        """
        self.dag = dag
        self.n_actions_dict = n_actions_dict
        self.n_iterations = n_iterations
        self.n_eval_episodes = n_eval_episodes
        self.alpha = alpha
        self.beta = beta
        self.memory_m = memory_m
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Structures de données
        self.q_tables: Dict[Tuple[int, int], Dict] = {}  # Q_u(θ, a)
        self.g_tables: Dict[Tuple[int, int], Dict] = {}  # G_u(h, a)
        self.policies: Dict[Tuple[int, int], MemoryPolicy] = {}  # σ_u(·|h)
        
        # Historique des (u, h) visités pour amélioration
        self.visited_uh = set()
        
        # Build successor map
        self._build_successor_map()
        
        # Initialiser politiques uniformes
        self._initialize_policies()
    
    def _build_successor_map(self):
        """Créer mapping (agent, time) → successor depuis le DAG"""
        self.successor_map = {}
        
        for stage in self.dag.stages:
            if stage.successor is not None:
                succ_stage = self.dag.stages[stage.successor]
                # DAG utilise agent 1-indexé, on convertit en 0-indexé
                self.successor_map[(stage.agent-1, stage.time)] = (succ_stage.agent-1, succ_stage.time)
            else:
                self.successor_map[(stage.agent-1, stage.time)] = None  # Terminal
    
    def _initialize_policies(self):
        """Initialiser politiques uniformes pour tous les substages"""
        for stage in self.dag.stages:
            u = (stage.agent - 1, stage.time)  # 0-indexed
            agent_id = stage.agent - 1
            self.policies[u] = MemoryPolicy(self.n_actions_dict[agent_id])
    
    def _extract_private_history(self, agent_id: int, theta: TypeTuple) -> Tuple:
        """Extraire historique privé h de l'agent depuis le type θ"""
        if not hasattr(theta, 'joint_histories'):
            return ()
        
        if agent_id >= len(theta.joint_histories):
            return ()
        
        agent_hist = theta.joint_histories[agent_id]
        obs_list = agent_hist.get('obs', [])
        acts_list = agent_hist.get('acts', [])
        
        # Limiter à memory_m dernières observations/actions
        if self.memory_m > 0:
            obs_list = obs_list[-self.memory_m:]
            acts_list = acts_list[-self.memory_m:]
        
        return (tuple(obs_list), tuple(acts_list))
    
    def _is_terminal(self, u: Tuple[int, int]) -> bool:
        """Vérifier si substage u est terminal"""
        return self.successor_map.get(u) is None
    
    def train(self, sim_factory, verbose: bool = True):
        """
        Boucle principale de Policy Iteration
        
        Args:
            sim_factory: Callable qui retourne un nouveau simulateur
            verbose: Afficher progression
        
        Returns:
            policies: Dictionnaire des politiques apprises
        """
        
        for iteration in range(self.n_iterations):
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Itération Policy Iteration {iteration+1}/{self.n_iterations}")
                print(f"{'='*60}")
            
            # Reset visited (u,h) pour cette itération
            self.visited_uh.clear()
            
            # PHASE 1 : Évaluation on-policy
            if verbose:
                print(f"Phase 1 : Évaluation on-policy ({self.n_eval_episodes} épisodes)...")
            
            self._evaluate_policy(sim_factory)
            
            if verbose:
                print(f"  → {len(self.visited_uh)} paires (u,h) visitées")
                print(f"  → Q-tables : {sum(len(qt) for qt in self.q_tables.values())} entrées")
                print(f"  → G-tables : {sum(len(gt) for gt in self.g_tables.values())} entrées")
            
            # PHASE 2 : Amélioration
            if verbose:
                print(f"Phase 2 : Amélioration de politique...")
            
            policy_changed = self._improve_policy(verbose=verbose)
            
            if verbose:
                print(f"  → Politique modifiée : {policy_changed}")
            
            # Convergence ?
            if not policy_changed:
                if verbose:
                    print(f"\n✅ Convergence atteinte à l'itération {iteration+1} !")
                break
        
        return self.policies
    
    def _evaluate_policy(self, sim_factory):
        """
        Évaluation on-policy avec backup de toutes les actions
        
        Pour chaque épisode :
            - Générer trajectoire avec σ courante (on-policy pur)
            - À chaque substage : backup TOUTES les actions
        """
        
        for episode in range(self.n_eval_episodes):
            
            # Créer nouveau simulateur
            sim = sim_factory()
            
            # Démarrer au premier substage
            theta = sim.reset()
            
            # Parcourir séquentiellement selon le simulateur
            done = False
            while not done:
                # Agent et temps courants
                agent_id = sim._curr_u.agent
                t = sim._curr_u.t
                u = (agent_id, t)
                
                # Extraire historique privé
                h = self._extract_private_history(agent_id, theta)
                
                # Marquer (u, h) comme visité
                self.visited_uh.add((u, h))
                
                # Action on-policy : a_on ~ σ(·|h)
                a_on = self.policies[u].sample(agent_id, h, self.rng)
                
                # Échantillonner transition avec action on-policy
                theta_star, r_star, done_star, info = sim.step(a_on)
                
                # ========================================
                # BACKUP TOUTES LES ACTIONS
                # ========================================
                
                for a in range(self.n_actions_dict[agent_id]):
                    
                    # Réutiliser transition on-policy si a = a_on
                    if a == a_on:
                        theta_a = theta_star
                        r_a = r_star
                        done_a = done_star
                    else:
                        # Échantillonner nouvelle transition pour cette action
                        # Créer nouveau simulateur et le restaurer à l'état θ
                        sim_alt = sim_factory()
                        self._restore_sim_state(sim_alt, theta)
                        theta_a, r_a, done_a, _ = sim_alt.step(a)
                    
                    # Calculer successeur en ESPÉRANCE
                    if not done_a:
                        # Agent suivant
                        u_next = (theta_a.prefix_actions.__len__(), theta_a.t) if hasattr(theta_a, 'prefix_actions') else None
                        if u_next is not None:
                            v_next = self._compute_expected_value(u_next, theta_a)
                        else:
                            v_next = 0.0
                    else:
                        v_next = 0.0
                    
                    # Erreur TD
                    if u not in self.q_tables:
                        self.q_tables[u] = {}
                    if theta not in self.q_tables[u]:
                        self.q_tables[u][theta] = np.zeros(self.n_actions_dict[agent_id])
                    
                    old_q = self.q_tables[u][theta][a]
                    delta = r_a + v_next - old_q
                    
                    # Mise à jour Q
                    self.q_tables[u][theta][a] = old_q + self.alpha * delta
                    
                    # Mise à jour G
                    if u not in self.g_tables:
                        self.g_tables[u] = {}
                    if h not in self.g_tables[u]:
                        self.g_tables[u][h] = np.zeros(self.n_actions_dict[agent_id])
                    
                    old_g = self.g_tables[u][h][a]
                    new_q = self.q_tables[u][theta][a]
                    self.g_tables[u][h][a] = old_g + self.beta * (new_q - old_g)
                
                # Avancer
                theta = theta_star
                done = done_star
    
    def _restore_sim_state(self, sim, theta: TypeTuple):
        """
        Restaurer l'état interne d'un simulateur à partir d'un type θ
        
        Args:
            sim: Simulateur à restaurer
            theta: Type contenant l'état complet
        """
        from declearn.core.sequential_env import SubStage
        
        # Restaurer l'état
        sim._curr_state = theta.state
        sim._curr_u = SubStage(agent=0, t=theta.t)
        
        # Restaurer les historiques
        if hasattr(theta, 'joint_histories'):
            for ag, hist in enumerate(theta.joint_histories):
                sim._obs_history[ag] = list(hist.get('obs', []))
                sim._act_history[ag] = list(hist.get('acts', []))
        
        # Restaurer prefix_actions
        sim._curr_joint_action = [None] * sim.n_agents
        if hasattr(theta, 'prefix_actions'):
            for ag, act in enumerate(theta.prefix_actions):
                if act is not None:
                    sim._curr_joint_action[ag] = act
        
        # Positionner le substage actuel (quel agent doit jouer)
        n_prefix = sum(1 for a in theta.prefix_actions if a is not None)
        sim._curr_u = SubStage(agent=n_prefix, t=theta.t)
    
    def _compute_expected_value(self, u: Tuple[int, int], theta: TypeTuple) -> float:
        """
        Calculer successeur en ESPÉRANCE : V(θ') = Σ_b σ(b|h') Q(θ', b)
        
        Args:
            u: Substage suivant
            theta: Type suivant
        
        Returns:
            Valeur espérée V
        """
        agent_id, t = u
        h = self._extract_private_history(agent_id, theta)
        
        # Si Q-table pas encore initialisée, retourner 0
        if u not in self.q_tables or theta not in self.q_tables[u]:
            return 0.0
        
        q_values = self.q_tables[u][theta]
        
        # Calculer V = Σ_b σ(b|h) Q(θ, b)
        v = 0.0
        probs = self.policies[u].probs(agent_id, h)
        for b in range(self.n_actions_dict[agent_id]):
            v += probs[b] * q_values[b]
        
        return v
    
    def _improve_policy(self, verbose: bool = False) -> bool:
        """
        Amélioration de politique : σ(h) ← argmax_a G(h, a)
        
        Returns:
            True si au moins une politique a changé
        """
        
        policy_changed = False
        n_changes = 0
        
        for (u, h) in self.visited_uh:
            
            if u not in self.g_tables or h not in self.g_tables[u]:
                continue
            
            g_values = self.g_tables[u][h]
            agent_id = u[0]
            
            # Ancienne action greedy
            old_action = self.policies[u].greedy(agent_id, h)
            
            # Nouvelle action greedy
            new_action = int(np.argmax(g_values))
            
            if old_action != new_action:
                policy_changed = True
                n_changes += 1
                
                if verbose and n_changes <= 5:  # Afficher les 5 premiers changements
                    print(f"    u={u}, h={h}: {old_action} → {new_action} (G={g_values})")
            
            # Mettre à jour politique (greedy déterministe via logits)
            # On met un grand logit pour l'action greedy, 0 pour les autres
            new_logits = np.ones(self.n_actions_dict[u[0]]) * (-100.0)  # Très négatif
            new_logits[new_action] = 100.0  # Très positif
            self.policies[u].logits[(u[0], h)] = new_logits
        
        if verbose and n_changes > 5:
            print(f"    ... et {n_changes - 5} autres changements")
        
        return policy_changed
    
    def act(self, agent_id: int, theta: TypeTuple, greedy: bool = True) -> int:
        """
        Choisir action pour agent_id avec type θ
        
        Args:
            agent_id: ID de l'agent
            theta: Type courant
            greedy: Si True, utilise politique greedy (déterministe)
        
        Returns:
            Action choisie
        """
        # Extraire t du theta
        if hasattr(theta, 't'):
            t = theta.t
        else:
            t = 0
        
        u = (agent_id, t)
        
        # Vérifier que politique existe
        if u not in self.policies:
            # Politique par défaut uniforme
            return self.rng.choice(self.n_actions_dict[agent_id])
        
        # Extraire historique privé
        h = self._extract_private_history(agent_id, theta)
        
        # Utiliser politique
        if greedy:
            return self.policies[u].greedy(agent_id, h)
        else:
            return self.policies[u].sample(agent_id, h, self.rng)
