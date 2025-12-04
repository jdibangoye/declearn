# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.

# TPI avec Exploration Guidée par Q-values

## Vue d'ensemble

**Problème identifié:** Les G-tables peuvent converger vers de mauvaises valeurs à cause d'un biais d'exploration en apprentissage multi-agent décentralisé.

**Solution:** Utiliser les Q-values pour **guider l'exploration** pendant l'entraînement, tout en continuant à apprendre les G-tables pour la politique finale décentralisée.

---

## Structures de Données

### 1. Q-tables: Q_u(θ, a)
- **Indexation:** Par type complet θ = (état, historiques de tous les agents)
- **Sémantique:** Valeur espérée de l'action a dans le type θ
- **Information:** Globale (connaît l'état de tous les agents)
- **Usage:** Guidage de l'exploration pendant l'entraînement

### 2. G-tables: G_u(h, a)  
- **Indexation:** Par historique privé h = historique de l'agent i uniquement
- **Sémantique:** Valeur espérée de l'action a sachant uniquement h
- **Information:** Locale (connaît seulement son propre historique)
- **Usage:** Politique exécutable de façon décentralisée

### 3. Politique π_u: π_u(a | h)
- **Indexation:** Par historique privé h
- **Sémantique:** Probabilité de choisir action a sachant h
- **Construction:** Softmax des G-values
- **Usage:** Exécution décentralisée

---

## Algorithme Complet

### Phase 1: ENTRAÎNEMENT (avec exploration guidée par Q)

```
Pour chaque épisode k = 1, 2, ..., N:
    
    # Initialisation
    sim = créer_simulateur()
    θ = sim.reset()  # Type initial complet
    
    # Boucle sur les agents (ordre du DAG)
    Pour chaque agent i dans l'ordre du DAG:
        
        # ============================================
        # CHOIX D'ACTION (exploration guidée par Q)
        # ============================================
        
        u = (i, t)  # Sous-paire (agent, temps)
        
        # Exploration ε-greedy
        ε_k = exploration_schedule(k)
        
        Si random() < ε_k:
            # Exploration uniforme
            a ← Uniforme(Actions_i)
        Sinon:
            # OPTION A: Utiliser Q (NEW - exploration guidée)
            Si use_q_values = True:
                a ← argmax_a' Q_u(θ, a')
            
            # OPTION B: Utiliser G (ORIGINAL - peut être biaisé)
            Sinon:
                h ← extraire_historique_privé(θ, i)
                a ← argmax_a' G_u(h, a')
        
        # ============================================
        # EXÉCUTION ET OBSERVATION
        # ============================================
        
        θ', r, done ← sim.step(a)
        
        # Échantillonner a' pour SARSA
        Si not done:
            i_next = prochain_agent(DAG)
            Si random() < ε_k:
                a' ← Uniforme(Actions_{i_next})
            Sinon:
                Si use_q_values:
                    a' ← argmax_a'' Q_{u'}(θ', a'')
                Sinon:
                    h' ← extraire_historique_privé(θ', i_next)
                    a' ← argmax_a'' G_{u'}(h', a'')
        
        # ============================================
        # MISES À JOUR (trois échelles de temps)
        # ============================================
        
        # 1. Mise à jour Q-table (échelle RAPIDE α_k)
        α_k = timescale.alpha_k(k)
        u' = (i_next, t')
        
        Q_u(θ, a) ← Q_u(θ, a) + α_k · [r + Q_{u'}(θ', a') - Q_u(θ, a)]
        
        # 2. Mise à jour G-table (échelle INTERMÉDIAIRE β_k)
        β_k = timescale.beta_k(k)
        h = extraire_historique_privé(θ, i)
        
        # G apprend en moyennant les Q-values
        q_value = Q_u(θ, a)
        G_u(h, a) ← G_u(h, a) + β_k · [q_value - G_u(h, a)]
        
        # 3. Mise à jour Politique (échelle LENTE γ_k)
        γ_k = timescale.gamma_k(k)
        
        # Construction du vecteur de G-values pour toutes les actions
        g_vector = [G_u(h, a') pour a' ∈ Actions_i]
        
        # Mirror descent pour mettre à jour π_u
        π_u.mirror_update(h, g_vector, γ_k)
        
        θ ← θ'
```

### Phase 2: EXÉCUTION (décentralisé, utilise seulement h)

```
# Après l'entraînement, pour exécuter la politique apprise

Pour chaque épisode:
    
    sim = créer_simulateur()
    observations_privées = sim.reset()
    
    Pour chaque agent i:
        
        # Agent i ne voit QUE son historique privé
        h_i = observations_privées[i]
        
        # Politique basée uniquement sur h_i (décentralisé !)
        a_i ← π_u(· | h_i).sample()
        
        # ou version greedy:
        a_i ← argmax_a G_u(h_i, a)
        
        observations_privées, r, done ← sim.step(a_i)
```

---

## Différences Clés entre ORIGINAL et MODIFIÉ

### ORIGINAL (TPI standard)
```
ENTRAÎNEMENT:
    h ← extraire_historique_privé(θ)
    a ← argmax_a G(h, a)          # Choix basé sur G
    Mise à jour Q, G, π
    
EXÉCUTION:
    h ← observations_privées
    a ← argmax_a G(h, a)          # Choix basé sur G
```

**Problème:** Si G converge vers de mauvaises valeurs (biais d'exploration), l'agent reste bloqué dans un optimum local.

### MODIFIÉ (TPI avec Q-guided exploration)
```
ENTRAÎNEMENT:
    a ← argmax_a Q(θ, a)          # Choix basé sur Q (info complète)
    Mise à jour Q, G, π
    ↓
    Grâce à meilleure exploration, G converge vers bonnes valeurs
    
EXÉCUTION:
    h ← observations_privées
    a ← argmax_a G(h, a)          # Choix basé sur G (maintenant correct!)
```

**Avantage:** Q guide l'exploration vers les bonnes régions, G apprend les bonnes valeurs, politique finale reste décentralisée.

---

## Exemple Concret: Recycling Problem

### Scénario
- 2 agents
- Actions: {searchbig=0, searchlittle=1, waitandrecharge=2}
- Récompenses:
  - π=(2,2): R=5 ⭐ OPTIMAL
  - π=(1,1): R=4
  - π=(0,1) ou (1,0): R=2
  - Autres: R=0

### Avec TPI ORIGINAL (G-guided)

**Épisodes 1-1000 (exploration initiale):**
```
Episode 100: π=(1,1) → R=4   [G₀(1)+=4, G₁(1)+=4]
Episode 200: π=(2,2) → R=5   [G₀(2)+=5, G₁(2)+=5]  ← rare !
Episode 300: π=(2,1) → R=2   [G₀(2)+=2, G₁(1)+=2]  ← biaise G(2) !
...
```

**Après 1000 épisodes:**
```
G₀(searchlittle=1) ≈ 4.0     (vu 730 fois avec R≈4)
G₀(waitandrecharge=2) ≈ 2.0  (vu 20 fois avec R=5, 100 fois avec R=2)
```

**Épisodes 1001-20000 (exploitation):**
```
Agent voit: G(1)=4.0 > G(2)=2.0
→ Agent choisit action=1 (searchlittle)
→ Les deux agents font searchlittle
→ π=(1,1) devient 73% des épisodes
→ G(2) continue à être sous-estimé
→ BLOQUÉ dans optimum local V=4 au lieu de V=5
```

### Avec TPI MODIFIÉ (Q-guided)

**Épisodes 1-1000 (exploration guidée par Q):**
```
Episode 100: θ₁=(état=0), Agent 0 choisit via Q
             Q(θ₁, 0) = ?
             Q(θ₁, 1) = 4   [après quelques essais]
             Q(θ₁, 2) = 5   [après quelques essais]
             → Choisit action=2 (waitandrecharge) ✅
```

**Q capture l'information jointe:**
```
Q((état=0, rien joué), waitandrecharge) = 5
    └─> Sait que si les deux font waitandrecharge → R=5

Q((état=0, rien joué), searchlittle) = 4
    └─> Sait que si les deux font searchlittle → R=4
```

**Résultat:**
```
Exploration guidée trouve π=(2,2) plus souvent
→ π=(2,2) devient 66% des épisodes
→ G(2) reçoit beaucoup d'échantillons avec R=5
→ G(2) converge vers 5.0 ✅
→ Politique finale π(a|h) est correcte
→ ATTEINT l'optimal V=5
```

---

## Relations entre Q, G, et π

### Flux d'information

```
Q-values (info globale)
    ↓ [moyennage sur types compatibles avec h]
G-values (info locale)
    ↓ [softmax / mirror descent]
Politique π(a|h)
```

### Formules

**G à partir de Q (mise à jour incrémentale):**
```
G_u(h, a) ← G_u(h, a) + β_k · [Q_u(θ, a) - G_u(h, a)]

où θ est tel que extraire_historique_privé(θ, i) = h
```

**Politique à partir de G (mirror descent):**
```
logits_u(h) ← logits_u(h) + γ_k · [g_vector - ⟨π_u(·|h), g_vector⟩]

π_u(a|h) = softmax(logits_u(h))[a]
```

---

## Garanties Théoriques

### Échelles de temps
```
α_k >> β_k >> γ_k >> 0

où α_k, β_k, γ_k → 0 quand k → ∞
```

**Conséquences:**
1. Q converge en premier (échelle rapide)
2. G converge ensuite vers moyenne des Q (échelle intermédiaire)
3. π converge en dernier vers softmax des G (échelle lente)

### Convergence avec Q-guided exploration

**Théorème (informel):**
Si Q converge vers Q*, et si exploration ε-greedy garantit que tous les types sont visités infiniment souvent, alors:

1. **Q converge:** Q_u(θ, a) → Q*_u(θ, a)
2. **G converge:** G_u(h, a) → E_θ[Q*_u(θ, a) | h] (moyenne correcte)
3. **π converge:** π_u(·|h) → politique optimale décentralisée

**Avec G-guided exploration:** G peut converger vers de mauvaises valeurs si exploration est biaisée (comme vu dans Recycling).

---

## Quand utiliser Q-guided vs G-guided?

### Utiliser Q-guided si:
- ✅ Problème de coordination (récompenses dépendent fortement des actions jointes)
- ✅ Faible nombre de types (Q-tables pas trop grandes)
- ✅ Entraînement centralisé possible (simulateur a accès à θ)

### Utiliser G-guided si:
- ✅ Problème bien conditionné (pas de piège d'exploration)
- ✅ Grand nombre de types (Q-tables trop grandes)
- ✅ Entraînement décentralisé requis dès le début

---

## Résumé en Une Phrase

**Q guide l'exploration pendant l'entraînement pour que G apprenne les bonnes valeurs, permettant une exécution décentralisée optimale.**

---

## Code Principal Modifié

```python
def act(self, agent_id, theta, greedy=False, k=0, use_q_values=False):
    """Choix d'action avec option Q-guided ou G-guided"""
    
    u = (agent_id, theta.t)
    
    # Exploration ε-greedy
    if not greedy:
        epsilon = self.exploration_schedule.epsilon(k)
        if random() < epsilon:
            return random_action()
    
    # NOUVEAU: Option Q-guided (bypass G-tables)
    if use_q_values and u in self.q_tables and theta in self.q_tables[u]:
        q_values = self.q_tables[u][theta]
        return argmax(q_values)
    
    # ORIGINAL: G-guided
    h = extract_private_history(agent_id, theta)
    return self.policies[u].greedy(agent_id, h)
```

Usage:
```python
# Entraînement avec Q-guided
action = tpi.act(agent_id, theta, greedy=True, use_q_values=True)

# Exécution avec G (décentralisé)
action = tpi.act(agent_id, theta, greedy=True, use_q_values=False)
```
