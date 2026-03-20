import os
import sys
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

# Add the src directory to sys.path
sys.path.append(os.path.join(os.getcwd(), "R-NaD", "src"))

from rnad import RNaDLearner, RNaDConfig

def test_rnad_model():
    config = RNaDConfig(hidden_size=128, num_blocks=1, num_heads=4)
    learner = RNaDLearner(state_dim=128, num_actions=400, config=config)
    
    key = jax.random.PRNGKey(42)
    learner.init(key)
    
    print("Model initialized successfully.")
    
    # Test different expert types
    state_types = {
        "combat": 0,
        "map": 1,
        "event": 2,
        "grid": 3,
        "hand": 4
    }
    
    for name, st_type in state_types.items():
        print(f"Testing expert type: {name} (idx: {st_type})")
        
        obs = {
            "global": jnp.zeros((1, 128)),
            "combat": jnp.zeros((1, 384)),
            "draw_bow": jnp.zeros((1, 100)),
            "discard_bow": jnp.zeros((1, 100)),
            "exhaust_bow": jnp.zeros((1, 100)),
            "master_bow": jnp.zeros((1, 100)),
            "map": jnp.zeros((1, 2048)),
            "event": jnp.zeros((1, 128)),
            "state_type": jnp.array([st_type], dtype=jnp.int32)
        }
        mask = jnp.ones((1, 400))
        
        try:
            logits, value = learner.network.apply(learner.params, key, obs, mask)
            print(f"  Success! Logits shape: {logits.shape}, Value shape: {value.shape}")
        except Exception as e:
            print(f"  Failed for {name}: {e}")
            # print(traceback.format_exc())

if __name__ == "__main__":
    test_rnad_model()
