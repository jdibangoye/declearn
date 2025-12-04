# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
"""Test de debug TPI avec TypeTuple hashable"""

import pytest
import numpy as np
from declearn.core.tabular_tpi import TabularTPI
from declearn.tests.test_tabular_tpi import MinimalDAG
from declearn.core.sa_schedules import ThreeTimeScale
from declearn.core.sequential_env import TypeTuple


def test_tpi_with_typetuple_hashable():
    """Test que TPI fonctionne avec TypeTuple comme clé hashable"""
    # Configuration TPI minimale
    dag = MinimalDAG(1)
    n_actions = {0: 2}
    ts = ThreeTimeScale(alpha=1.0, beta=0.1, gamma=0.01)
    rng = np.random.default_rng(42)

    tpi = TabularTPI(dag, n_actions, ts, 1, rng)

    # Test échantillon simple avec TypeTuple
    theta = TypeTuple(
        state=1, t=0, 
        joint_histories=[{'obs': [1], 'acts': [-1]}], 
        prefix_actions=[]
    )
    theta_next = TypeTuple(
        state=2, t=1, 
        joint_histories=[{'obs': [1,2], 'acts': [-1,1]}], 
        prefix_actions=[]
    )
    
    sample = {
        0: {
            "theta": theta,
            "a": 1, "r": 1.0, 
            "theta_next": theta_next,
            "h": (1,), "h_next": (2,), "a_next": 0, "u": 0
        }
    }
    
    # Test que le step fonctionne sans lever d'exception
    tpi.step(sample, k=0)
    
    # Vérifications de base - utiliser nouvelle structure u=(agent, t)
    u_pair = (0, 0)  # agent=0, t=0
    assert u_pair in tpi.q_tables
    assert theta in tpi.q_tables[u_pair]  # TypeTuple est maintenant hashable
    
    # Test que l'action peut être calculée
    action = tpi.act(0, theta, greedy=True)
    assert action in [0, 1]  # action valide


def test_typetuple_hashable():
    """Test spécifique que TypeTuple est hashable et fonctionne comme clé"""
    
    # Créer deux TypeTuple identiques
    tt1 = TypeTuple(
        state=1, t=0,
        joint_histories=[{'obs': [1], 'acts': [0]}],
        prefix_actions=[]
    )
    tt2 = TypeTuple(
        state=1, t=0, 
        joint_histories=[{'obs': [1], 'acts': [0]}],
        prefix_actions=[]
    )
    
    # Test qu'ils sont égaux
    assert tt1 == tt2
    
    # Test qu'ils ont le même hash
    assert hash(tt1) == hash(tt2)
    
    # Test qu'ils peuvent être utilisés comme clés de dictionnaire
    test_dict = {tt1: "value1"}
    assert test_dict[tt2] == "value1"  # devrait marcher car tt1 == tt2
    
    test_dict[tt2] = "value2"
    assert len(test_dict) == 1  # pas de nouvelle entrée car même clé