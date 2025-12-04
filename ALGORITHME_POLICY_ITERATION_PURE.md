# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.

# Type-based Policy Iteration (TPI) - Version Policy Iteration Pure

## Principe

Au lieu de mélanger exploration et exploitation (GLIE), on alterne entre :
1. **Évaluation on-policy** : Évaluer la politique courante σ avec échantillons purs (pas d'exploration)
   - **Clé** : Backup **toutes les actions** a ∈ A, pas seulement celle prise
   - Trajectoire suivie avec action on-policy `a_on ~ σ(·|h)`
   - Mais on simule aussi les autres actions pour mettre à jour leurs Q-values
2. **Amélioration** : Améliorer σ basé sur les Q-values/G-values apprises

C'est l'approche classique de **Policy Iteration** appliquée à Dec-POMDP avec types.

## Pourquoi Backup Toutes les Actions ?

**Problème sans backup complet** :
- Si on évalue seulement `a_on ~ σ(h)`, on n'apprend que Q(θ, a_on)
- À l'amélioration, argmax_a G(h,a) ne peut pas choisir une meilleure action car les autres Q sont inconnues
- Politique reste bloquée !

**Solution avec backup complet** :
- Action on-policy `a_on` génère la trajectoire principale
- Pour chaque `a ∈ A` :
  - Si `a = a_on` : utiliser transition déjà échantillonnée (θ*, r*)
  - Sinon : échantillonner nouvelle transition (θ_a', r_a) ~ G(·|θ, a)
  - Mettre à jour Q(θ, a) avec successeur en espérance V(θ') = Σ_b σ(b|h') Q(θ', b)
- Résultat : Toutes les Q-values sont apprises → amélioration possible

---

## Algorithme Principal (TPI-AllActions)

```
INITIALISATION:
    σ_u(·|h) = Uniforme pour tous u, h
    Q_u(θ, a) = 0 pour tous u, θ, a
    G_u(h, a) = 0 pour tous u, h, a
    
RÉPÉTER jusqu'à convergence:
    
    # ========================================
    # PHASE 1: ÉVALUATION ON-POLICY
    # ========================================
    
    Pour episode = 1, ..., E:
        
        # Démarrer au substage terminal inversé (racine du DAG)
        u ← u_terminal
        θ ~ distribution_initiale()
        
        # Appel récursif de l'évaluation
        TrialEval(u, θ)
    
    # ========================================
    # PHASE 2: AMÉLIORATION DE POLITIQUE
    # ========================================
    
    Δ ← False
    
    Pour chaque (u, h) visité durant évaluation:
        
        # Nouvelle politique gloutonne basée sur G
        μ_u(h) ← argmax_a G_u(h, a)
        
        # Détecter changement
        Si μ_u(h) ≠ σ_u(h):
            Δ ← True
        
        # Mettre à jour politique
        σ_u(h) ← μ_u(h)
    
JUSQU'À ¬Δ (pas de changement)


# ========================================
# PROCÉDURE: TrialEval(u, θ)
# ========================================

Procédure TrialEval(u, θ):
    
    # Cas terminal : fin de récursion
    Si u = u_terminal:
        RETOURNER
    
    # Extraire historique privé
    h ← Priv_u(θ)
    
    # Action on-policy (pour générer trajectoire)
    a_on ~ σ_u(·|h)
    
    # Successeur suivant
    u' ← Succ(u)
    
    # Échantillonner transition avec action on-policy
    (θ*, r*) ~ Gen(·|θ, a_on)
    
    # Récursion : évaluer le reste de la trajectoire
    TrialEval(u', θ*)
    
    # ========================================
    # BACKUP POUR TOUTES LES ACTIONS
    # ========================================
    
    Pour chaque a ∈ Actions_u:
        
        # Réutiliser transition on-policy si a = a_on
        Si a = a_on:
            θ_a' ← θ*
            r_a ← r*
        Sinon:
            # Échantillonner nouvelle transition pour cette action
            (θ_a', r_a) ~ Gen(·|θ, a)
        
        # Extraire historique privé du successeur
        h_a' ← Priv_{u'}(θ_a')
        
        # Successeur en ESPÉRANCE (au lieu d'échantillonner)
        V_{u'} ← Σ_{b ∈ Actions_{u'}} σ_{u'}(b|h_a') · Q_{u'}(θ_a', b)
        
        # Erreur TD
        δ_a ← r_a + V_{u'} - Q_u(θ, a)
        
        # Mise à jour Q
        Q_u(θ, a) ← Q_u(θ, a) + α · δ_a
        
        # Mise à jour G (moyenne des Q pour ce h)
        G_u(h, a) ← G_u(h, a) + β · [Q_u(θ, a) - G_u(h, a)]
```

---

## Points Clés de l'Algorithme

### 1. Structure Récursive
- `TrialEval(u, θ)` suit l'ordre du DAG
- Évalue récursivement du substage u jusqu'au terminal
- Permet de calculer les valeurs par backward induction

### 2. Backup Toutes Actions
```
Pour chaque a ∈ A:
    Si a = a_on : réutiliser (θ*, r*) déjà échantillonné
    Sinon : échantillonner nouvelle transition (θ_a', r_a)
    
    Mettre à jour Q(θ, a) et G(h, a)
```

**Avantage** : On apprend Q pour TOUTES les actions, pas seulement celle prise
→ Amélioration possible à la prochaine itération

### 3. Successeur en Espérance
```
V(θ') = Σ_b σ(b|h') · Q(θ', b)
```

Au lieu de :
```
a' ~ σ(·|h')
V(θ') = Q(θ', a')  # Estimateur biaisé !
```

**Avantage** : Estimateur non biaisé de la valeur du successeur

### 4. Two-Timescale
- α pour Q-tables (rapide)
- β pour G-tables (lent) avec β << α
- Pas de mise à jour de politique pendant évaluation (policy iteration pure)

---

## Avantages de cette Approche

### 1. Théoriquement Fondée
- ✅ **Policy Iteration classique** : Garanties de convergence connues
- ✅ **On-policy pur** : Trajectoires générées avec σ courante (pas d'ε-greedy)
- ✅ **Décentralisé** : Politique σ(a|h) basée uniquement sur h
- ✅ **Backup complet** : Toutes les actions évaluées → amélioration possible

### 2. Robuste à Tout Horizon
- ✅ Fonctionne à T=1, T=2, ..., T=∞
- ✅ Pas de "triche" avec θ complet pendant évaluation
- ✅ Structure claire : évaluation puis amélioration
- ✅ Récursion suit naturellement le DAG

### 3. Pas de Biais d'Exploration
- ✅ Chaque itération évalue **exactement** la politique courante σ
- ✅ Amélioration basée sur Q-values **de toutes les actions**
- ✅ Pas de problème de GLIE qui reste bloqué
- ✅ Successeur en espérance (non biaisé)

---

## Comparaison avec Approches Précédentes

### 1. Approche Q-guided (avec GLIE)
```
Problèmes:
❌ Utilise θ complet pendant entraînement (pas décentralisé)
❌ GLIE peut rester bloqué dans optimum local
❌ Pas de garanties théoriques claires à T>1
❌ G-tables biaisées par exploration déséquilibrée

Avantages:
✅ Simple à implémenter
✅ Marche bien à T=1 sur certains benchmarks
```

### 2. Policy Iteration sans backup complet (défaut)
```
Problèmes:
❌ Évalue seulement Q(θ, a_on) pour action prise
❌ Ne peut pas améliorer : argmax_a G(h,a) manque d'info
❌ Politique reste bloquée à l'initialisation

Avantages:
✅ Économe en samples (1 transition par substage)
```

### 3. Policy Iteration avec backup complet (TPI-AllActions)
```
Avantages:
✅ Complètement décentralisée (σ utilise seulement h)
✅ Garanties théoriques de Policy Iteration
✅ Fonctionne à tout horizon T
✅ Backup complet → toutes Q-values apprises
✅ Successeur en espérance → estimateur non biaisé
✅ Pas de biais d'exploration

Défis:
⚠️  |A| transitions échantillonnées par substage (vs 1 sans backup)
⚠️  Besoin de E épisodes par itération d'évaluation
⚠️  Convergence peut être lente (plusieurs itérations PI)
⚠️  Risque optimum local (mais intrinsèque à Policy Iteration)
```

---

## Exemple Concret : Recycling T=1

### Configuration
- 2 agents, horizon T=1
- Actions : {searchlittle=0, searchextensive=1, waitandrecharge=2}
- État initial : s₀ avec P(s₀)=1, R(s₀, (wait,wait))=5, R(s₀, (search,search))=4

### Itération 0 : Initialisation
```
σ_u(a|h) = Uniforme(1/3, 1/3, 1/3) pour tous u, h
Q_u(θ,a) = 0
G_u(h,a) = 0
```

### Itération 1 : Évaluation

Episode 1 :
```
θ = type_initial
h = ()
a_on ~ σ(·|()) = action 1 (searchextensive) échantillonnée

Pour a=0 (searchlittle):
    (θ',r) ~ Gen(·|θ,0) → r=4
    V' = Σ_b σ(b|h') Q(θ',b) = 0 (car agent suivant)
    δ = 4 + 0 - 0 = 4
    Q(θ,0) ← 0 + α·4
    G((),0) ← 0 + β·Q(θ,0)

Pour a=1 (searchextensive) [on-policy]:
    Réutiliser (θ*,r*) déjà échantillonné → r=4
    δ = 4 + 0 - 0 = 4
    Q(θ,1) ← 0 + α·4
    G((),1) ← 0 + β·Q(θ,1)

Pour a=2 (waitandrecharge):
    (θ',r) ~ Gen(·|θ,2) → r dépend de ce que l'autre agent fait
    ...
```

Après E épisodes : Toutes les Q(θ,a) sont apprises !

### Itération 1 : Amélioration
```
Pour h=():
    G((),0) ≈ 4.0
    G((),1) ≈ 3.5
    G((),2) ≈ valeur moyenne (dépend des samples)
    
    μ(()) = argmax_a G((),a) = 0 (searchlittle)
    
    Changement : σ(()) = Uniforme → μ(()) = searchlittle
    Δ = True
```

### Itération 2 : Évaluation avec nouvelle politique

Maintenant σ(()) = searchlittle (deterministe)
```
Tous les épisodes prennent a_on = searchlittle
Mais on backup quand même les 3 actions !

Pour a=2 (waitandrecharge):
    Plus de samples où autres agents font wait aussi
    G((),2) converge vers 5.0
```

### Itération 2 : Amélioration
```
G((),0) ≈ 4.0
G((),1) ≈ 3.5  
G((),2) ≈ 5.0  ← Maintenant c'est le meilleur !

μ(()) = argmax_a G((),a) = 2 (waitandrecharge)

Changement : σ(()) = searchlittle → μ(()) = waitandrecharge
Δ = True
```

### Itération 3 : Convergence
```
σ(()) = waitandrecharge pour les 2 agents
Tous épisodes : (wait, wait) → r=5
G((),2) = 5.0 reste optimal
Δ = False → FIN
```

---

## Détails d'Implémentation

### Structure de Boucle

```python
class TabularPolicyIterationAllActions:
    
    def __init__(self, dag, n_actions, n_iterations=10, n_eval_episodes=5000, alpha=0.1, beta=0.01):
        self.dag = dag
        self.n_actions = n_actions
        self.n_iterations = n_iterations
        self.n_eval_episodes = n_eval_episodes
        self.alpha = alpha  # Learning rate pour Q
        self.beta = beta    # Learning rate pour G (β << α)
        
        # Structures
        self.q_tables = {}  # Q_u(θ, a)
        self.g_tables = {}  # G_u(h, a)
        self.policies = {}  # σ_u(a|h) - déterministe après amélioration
        
        # Initialiser politiques uniformes
        for u in all_substages(dag):
            self.policies[u] = UniformPolicy(n_actions[u])
    
    def train(self, env_factory):
        """Boucle principale de Policy Iteration"""
        
        for iteration in range(self.n_iterations):
            
            print(f"Itération {iteration+1}/{self.n_iterations}")
            
            # PHASE 1: Évaluation on-policy
            self.evaluate_policy(env_factory)
            
            # PHASE 2: Amélioration
            policy_changed = self.improve_policy()
            
            # Convergence ?
            if not policy_changed:
                print("Politique convergée!")
                break
        
        return self.policies
    
    def evaluate_policy(self, env_factory):
        """Évaluation on-policy avec backup de toutes les actions"""
        
        # NE PAS réinitialiser Q et G (accumulation across iterations)
        # Initialiser si première fois
        if not self.q_tables:
            for u in all_substages(self.dag):
                self.q_tables[u] = {}
                self.g_tables[u] = {}
        
        for episode in range(self.n_eval_episodes):
            
            # Générer état initial
            env = env_factory()
            theta = env.reset()
            
            # Démarrer évaluation récursive
            self._trial_eval(env, u_terminal, theta)
    
    def _trial_eval(self, env, u, theta):
        """Évaluation récursive avec backup de toutes les actions
        
        Args:
            env: Environnement (pour échantillonner transitions)
            u: Substage courant
            theta: Type courant
        """
        
        # Cas terminal
        if self._is_terminal(u):
            return
        
        # Extraire historique privé
        agent_id = self._get_agent(u)
        h = self._extract_history(theta, agent_id)
        
        # Action on-policy
        a_on = self.policies[u].sample(h)
        
        # Substage suivant
        u_next = self._get_successor(u)
        
        # Échantillonner transition avec action on-policy
        theta_star, r_star, done = env.step_from_state(theta, a_on)
        
        # Récursion : évaluer le reste de la trajectoire
        if not done:
            self._trial_eval(env, u_next, theta_star)
        
        # ========================================
        # BACKUP TOUTES LES ACTIONS
        # ========================================
        
        for a in range(self.n_actions[agent_id]):
            
            # Réutiliser transition on-policy si a = a_on
            if a == a_on:
                theta_a = theta_star
                r_a = r_star
            else:
                # Échantillonner nouvelle transition pour cette action
                theta_a, r_a, done_a = env.step_from_state(theta, a)
            
            # Successeur en ESPÉRANCE
            if not done:
                h_a = self._extract_history(theta_a, self._get_agent(u_next))
                
                # V = Σ_b σ(b|h') Q(θ',b)
                v_next = 0.0
                if u_next in self.q_tables and theta_a in self.q_tables[u_next]:
                    for b in range(self.n_actions[self._get_agent(u_next)]):
                        prob_b = self.policies[u_next].get_prob(h_a, b)
                        q_b = self.q_tables[u_next][theta_a][b]
                        v_next += prob_b * q_b
            else:
                v_next = 0.0
            
            # Erreur TD
            if u not in self.q_tables:
                self.q_tables[u] = {}
            if theta not in self.q_tables[u]:
                self.q_tables[u][theta] = np.zeros(self.n_actions[agent_id])
            
            old_q = self.q_tables[u][theta][a]
            delta = r_a + v_next - old_q
            
            # Mise à jour Q
            self.q_tables[u][theta][a] = old_q + self.alpha * delta
            
            # Mise à jour G
            if u not in self.g_tables:
                self.g_tables[u] = {}
            if h not in self.g_tables[u]:
                self.g_tables[u][h] = np.zeros(self.n_actions[agent_id])
            
            old_g = self.g_tables[u][h][a]
            new_q = self.q_tables[u][theta][a]
            self.g_tables[u][h][a] = old_g + self.beta * (new_q - old_g)
    
    def improve_policy(self):
        """Amélioration de politique basée sur G-tables"""
        
        policy_changed = False
        
        for u in all_substages(self.dag):
            if u not in self.g_tables:
                continue
            
            for h, g_values in self.g_tables[u].items():
                
                # Ancienne action greedy
                old_action = np.argmax(self.policies[u].get_probs(h))
                
                # Nouvelle action greedy
                new_action = np.argmax(g_values)
                
                if old_action != new_action:
                    policy_changed = True
                
                # Mettre à jour politique (greedy ou softmax)
                self.policies[u].set_greedy(h, new_action)
                # OU avec softmax:
                # self.policies[u].set_softmax(h, g_values, temperature=0.1)
        
        return policy_changed
```

---

## Variantes Possibles

### Variante 1: Avec Exploration Initiale

```python
for iteration in range(n_iterations):
    
    # Décroire exploration au fil des itérations
    epsilon = max(0.1, 1.0 - iteration / n_iterations)
    
    # Évaluation avec un peu d'exploration
    if iteration < n_iterations // 2:
        self.evaluate_policy(env_factory, epsilon=epsilon)
    else:
        self.evaluate_policy(env_factory, epsilon=0.0)  # Pure exploitation
    
    self.improve_policy()
```

### Variante 2: Soft Policy Iteration

```python
def improve_policy(self, temperature=1.0):
    """Amélioration avec softmax au lieu de greedy"""
    
    for u, g_dict in self.g_tables.items():
        for h, g_values in g_dict.items():
            # Softmax au lieu de argmax
            probs = softmax(g_values / temperature)
            self.policies[u].set_probs(h, probs)
    
    # Décroire température au fil des itérations
    temperature *= 0.9
```

### Variante 3: Batch Policy Iteration

```python
# Au lieu de N_iterations × N_eval_episodes
# Faire 1 grosse collecte puis multiples améliorations

# Collecter beaucoup de données une fois
trajectories = collect_many_trajectories(N_total_episodes)

for iteration in range(n_iterations):
    
    # Sous-échantillonner trajectoires
    batch = random_sample(trajectories, batch_size)
    
    # Évaluer sur ce batch
    self.evaluate_on_batch(batch)
    
    # Améliorer
    self.improve_policy()
```

---

## Résumé

### L'idée centrale

**Au lieu de:**
```
Apprendre Q, G, σ simultanément avec GLIE décroissant (ε-greedy)
```

**On fait:**
```
RÉPÉTER:
    1. ÉVALUATION: Générer E épisodes avec σ courante (on-policy pur)
                   Pour chaque substage visité:
                       - Action on-policy: a_on ~ σ(·|h)
                       - Backup TOUTES les actions a ∈ A
                       - Successeur en espérance: V' = Σ_b σ(b|h') Q(θ',b)
                   
    2. AMÉLIORATION: Pour chaque (u,h) visité:
                     σ(h) ← argmax_a G(h,a)
                     
JUSQU'À convergence (Δ = False)
```

### Pourquoi c'est mieux théoriquement

1. **Décentralisé** : σ(a|h) n'utilise que h (pas θ complet)
2. **On-policy** : Trajectoires générées avec σ courante (pas d'exploration artificielle)
3. **Backup complet** : Toutes les Q(θ,a) apprises → amélioration possible
4. **Successeur non biaisé** : Espérance au lieu d'échantillon unique
5. **Policy Iteration** : Garanties de convergence classiques
6. **Fonctionne à tout T** : Pas de hack spécifique à T=1

### Le prix à payer

- **|A| fois plus de samples** : Backup de |A| actions au lieu de 1
- **Plusieurs itérations** : Besoin de E épisodes × N_iter
- **Convergence locale** : Comme tout Policy Iteration (mais intrinsèque)

### Mais on gagne

- ✅ Garanties théoriques solides
- ✅ Pas de biais G dû à exploration déséquilibrée
- ✅ Fonctionne à tout horizon T
- ✅ Complètement décentralisé

---

## Prochaines Étapes

1. **Implémenter** TPI-AllActions dans `declearn/core/policy_iteration_allactions.py`
2. **Tester** sur Recycling T=1 (devrait converger vers V=5.0)
3. **Comparer** avec approche Q-guided
4. **Valider** sur T=2, T=3, ... (test de robustesse)
5. **Benchmarks** : BroadcastChannel, DecTiger, BoxPushing, GridSmall

Voulez-vous que je procède à l'implémentation ?
