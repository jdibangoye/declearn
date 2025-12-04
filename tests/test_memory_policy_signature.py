# Ce projet appartient a Jesus-Christ, pour la gloire de Dieu et le service de la verite.
from declearn.core.memory_policy import MemoryPolicy
import inspect

# Inspecter la signature
sig = inspect.signature(MemoryPolicy.__init__)
print(f"MemoryPolicy signature: {sig}")

# Test d'instantiation
import numpy as np
rng = np.random.default_rng(42)

# Essayer différentes signatures
try:
    policy1 = MemoryPolicy(2, 1, rng)
    print("✅ Signature (n_actions, memory_m, rng) fonctionne")
except Exception as e:
    print(f"❌ Signature (n_actions, memory_m, rng): {e}")

try:
    policy2 = MemoryPolicy(2, 1)
    print("✅ Signature (n_actions, memory_m) fonctionne")
except Exception as e:
    print(f"❌ Signature (n_actions, memory_m): {e}")

try:
    policy3 = MemoryPolicy(2)
    print("✅ Signature (n_actions) fonctionne")
except Exception as e:
    print(f"❌ Signature (n_actions): {e}")