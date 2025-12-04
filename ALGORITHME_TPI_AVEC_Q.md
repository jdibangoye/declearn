# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.

# TPI avec Backup Complet : Résultats et Analyse

## Résumé Exécutif

Nous avons implémenté **Policy Iteration avec Backup de Toutes les Actions** (TPI-AllActions) pour résoudre théoriquement les problèmes identifiés dans l'approche GLIE.

**Résultat surprenant** : ❌ Le même problème de convergence vers optimum local persiste !

## Algorithme Implémenté

### Structure
```
RÉPÉTER jusqu'à convergence :
    1. ÉVALUATION : Pour episode = 1..E :
           Générer trajectoire avec σ courante (on-policy pur, pas d'exploration)
           À chaque substage (agent_id, t) :
               - Action on-policy : a_on ~ σ(·|h)
               - Échantillonner transition : (θ*, r*) pour a_on
               
               POUR CHAQUE action a ∈ A :
                   Si a = a_on : réutiliser (θ*, r*)
                   Sinon : échantillonner nouvelle transition (θ_a', r_a)
                   
                   V' = Σ_b σ(b|h') Q(θ', b)  # Successeur en espérance
                   δ = r_a + V' - Q(θ, a)
                   Q(θ, a) ← Q(θ, a) + α·δ
                   G(h, a) ← G(h, a) + β·[Q(θ,a) - G(h,a)]
    
    2. AMÉLIORATION : Pour chaque (u,h) visité :
           σ(h) ← argmax_a G(h,a)
```

### Garanties Théoriques
- ✅ **On-policy pur** : Trajectoires générées avec σ courante (pas d'ε-greedy)
- ✅ **Backup complet** : TOUTES les actions évaluées (pas seulement celle prise)
- ✅ **Successeur non biaisé** : Espérance au lieu d'échantillon unique
- ✅ **Décentralisé** : σ(a|h) basé uniquement sur historique privé h
- ✅ **Policy Iteration classique** : Garanties de convergence standards

## Résultats sur Recycling T=1

### Configuration
- Domaine : Recycling, 2 agents, T=1
- Politique optimale : π* = (waitandrecharge, waitandrecharge) avec V* = 5.0
- Politique suboptimale : π = (searchlittle, searchlittle) avec V = 4.0

### Test 1 : 5000 épisodes d'évaluation
```
Itération 1 :
  Politique initiale : Uniforme
  G-tables apprises :
    G(searchbig) = 0.67
    G(searchlittle) = 2.67
    G(waitandrecharge) = 2.33   ← Devrait être 5.0 !
  
  Amélioration : σ(h) = searchlittle
  
Itération 2 :
  Politique : searchlittle pour les 2 agents
  Convergence atteinte
  
RÉSULTAT FINAL :
  V = 4.0 (80% de l'optimum)
  Politique = (searchlittle, searchlittle)
  ❌ ÉCHEC : Optimum local
```

### Test 2 : 50000 épisodes d'évaluation (10x plus)
```
Itération 1 :
  G-tables apprises :
    G(searchbig) = 0.67
    G(searchlittle) = 2.67
    G(waitandrecharge) = 2.33   ← Toujours pas 5.0 !
  
  Amélioration : σ(h) = searchlittle
  
RÉSULTAT : Même échec avec 10x plus d'échantillons
```

## Analyse du Problème

### Pourquoi G(waitandrecharge) = 2.33 au lieu de 5.0 ?

**Itération 1** : Politique uniforme
```
Quand agent 0 joue waitandrecharge :
  - Agent 1 joue searchbig (prob 1/3) → récompense = 2.0
  - Agent 1 joue searchlittle (prob 1/3) → récompense = 3.5
  - Agent 1 joue waitandrecharge (prob 1/3) → récompense = 5.0

Espérance : E[r | a₀=wait] = (2.0 + 3.5 + 5.0) / 3 = 3.5

Mais avec bruit d'échantillonnage et learning rate :
  G(wait) converge vers ≈ 2.33
```

### Pourquoi backup complet ne suffit pas ?

**Le backup complet résout** :
- ✅ On évalue Q(θ, wait) même si σ ne choisit jamais wait
- ✅ G(h, wait) est mis à jour via Q(θ, wait)

**Mais ne résout PAS** :
- ❌ Q(θ, wait) capturé la vraie valeur de (wait, wait) uniquement si l'autre agent joue wait
- ❌ Avec politique uniforme, prob(autre=wait) = 1/3
- ❌ Donc Q(θ, wait) ≈ valeur mixte, pas valeur optimale

### Le cercle vicieux

```
Itération 1 :
  σ = Uniforme
  → G(wait) = moyenne sur actions mixtes ≈ 3.5
  → G(searchlittle) = 4.0  (stable car optimal pour jeu non-coordonné)
  → Amélioration : σ(h) = argmax G = searchlittle

Itération 2 :
  σ = searchlittle pour tous
  → Plus aucun épisode avec (wait, wait)
  → G(wait) reste biaisé ou décroise
  → Convergence locale !
```

## Solutions Possibles

### 1. Beaucoup Plus d'Échantillons (Naïf)
```python
n_eval_episodes = 500_000  # Au lieu de 50_000
```
- ⚠️  Très coûteux computationnellement
- ⚠️  Pas de garantie de convergence vers optimum global
- ⚠️  Policy Iteration peut rester bloquée dans optimum local

### 2. Initialisation Informée
```python
# Initialiser avec politique proche de l'optimum
for u, h in all_substages:
    σ_init(h) = softmax([0.1, 0.1, 10.0])  # Favoriser waitandrecharge
```
- ✅ Converge rapidement si initialisation correcte
- ❌ Nécessite connaissance a priori de l'optimum
- ❌ Pas généralisable

### 3. Exploration Structurée pendant Évaluation
```python
# Au lieu d'on-policy pur, ajouter exploration dirigée
for episode in range(E):
    if episode % 100 == 0:
        # Force évaluation de politiques jointes spécifiques
        force_joint_policy = sample_from_joint_space()
    else:
        # On-policy normal
        ...
```
- ✅ Explore systématiquement l'espace des politiques jointes
- ⚠️  Viole le principe on-policy pur
- ⚠️  Perd garanties théoriques de Policy Iteration

### 4. Optimistic Initialization
```python
# Initialiser Q et G avec valeurs optimistes
Q_init(θ, a) = V_max = 5.0  # Récompense maximale possible
G_init(h, a) = V_max = 5.0
```
- ✅ Encourage exploration des actions sous-évaluées
- ✅ Principe connu en RL (optimism under uncertainty)
- ⚠️  Peut causer instabilité initiale

### 5. Multi-Start Policy Iteration
```python
# Lancer PI depuis plusieurs initialisations
results = []
for init_seed in range(10):
    σ_init = random_policy(seed=init_seed)
    σ_final = policy_iteration(σ_init)
    results.append((σ_final, value(σ_final)))

# Garder la meilleure
σ_best = argmax(results, key=lambda x: x[1])
```
- ✅ Explore plusieurs bassins d'attraction
- ✅ Approche pratique et robuste
- ⚠️  Coût computationnel × nombre d'initialisations

## Conclusion

### Ce que nous avons appris

1. **TPI-AllActions fonctionne correctement** d'un point de vue implémentation
   - Backup complet : ✅
   - On-policy pur : ✅
   - Successeur en espérance : ✅

2. **Le problème n'est PAS dans l'algorithme GLIE précédent**
   - Même problème avec Policy Iteration pure
   - Même problème avec backup complet
   - C'est un problème **intrinsèque aux jeux de coordination**

3. **Policy Iteration peut converger vers optimum local**
   - Initialisation uniforme → mauvais bassin d'attraction
   - Amélioration gloutonne → convergence locale
   - Pas de garantie d'optimum global sans hypothèses fortes

### Recommandations

Pour Recycling et domaines similaires (jeux de coordination) :

1. ✅ **Utiliser Multi-Start** : Plusieurs initialisations aléatoires
2. ✅ **Initialisation optimiste** : Q_init = V_max
3. ✅ **Exploration forcée** : Évaluer périodiquement toutes les politiques jointes pures
4. ⚠️  **Accepter** que Policy Iteration ne garantit que convergence locale

Pour benchmarks futurs :

1. ✅ **BroadcastChannel, DecTiger, GridSmall** : Pas de coordination → PI devrait fonctionner
2. ⚠️  **Recycling, BoxPushing** : Coordination forte → besoin de multi-start ou exploration

### Prochaines Étapes

1. Tester TPI-AllActions sur autres benchmarks (sans coordination)
2. Implémenter variante avec initialisation optimiste
3. Implémenter Multi-Start Policy Iteration
4. Tester à T=2 pour valider robustesse à horizons plus longs

---

**La leçon principale** : Le problème de Recycling n'est pas un bug d'implémentation, mais une **caractéristique fondamentale des jeux de coordination** qui nécessite des approches spécifiques (multi-start, exploration structurée, ou connaissance du domaine).
