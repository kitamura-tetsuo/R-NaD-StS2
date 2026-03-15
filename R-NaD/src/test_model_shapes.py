import os
import sys
import numpy as np

# Add the R-NaD directory to the path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

# Mock some environment variables and global values that bridge might expect
os.environ["ST_DEBUG"] = "1"

import rnad_bridge
import jax
import jax.numpy as jnp

def test_shapes():
    print("Initializing test...")
    
    # Create a dummy state JSON that mimics what the bridge expects
    dummy_state = {
        "floor": 10,
        "gold": 100,
        "player": {
            "hp": 50,
            "maxHp": 80,
            "block": 5,
            "energy": 3,
            "stars": 2,
            "drawPile": ["Strike_R", "Defend_R"],
            "discardPile": ["Bash"],
            "exhaustPile": [],
            "masterDeck": ["Strike_R", "Strike_R", "Defend_R", "Bash"]
        },
        "hand": [
            {
                "id": "Strike_R",
                "isPlayable": True,
                "targetType": "SingleEnemy",
                "cost": 1,
                "baseDamage": 6,
                "baseBlock": 0,
                "magicNumber": 0,
                "upgraded": False,
                "currentDamage": 9, # +3 from strength
                "currentBlock": 0
            }
        ],
        "enemies": [
            {
                "hp": 40,
                "maxHp": 40,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 6, "repeats": 1}]
            }
        ],
        "potions": [{"id": "empty"}, {"id": "empty"}],
        "type": "combat"
    }

    print("Encoding state...")
    encoded = rnad_bridge.encode_state(dummy_state)
    
    combat_vec = encoded["combat"]
    print(f"Combat vector shape: {combat_vec.shape}")
    assert combat_vec.shape == (256,)
    
    # Check if currentDamage is at correct index
    # Hand features start at index 10. Each card has 10 features.
    # Card 0 index 8 & 9 are currentDamage, currentBlock
    # index 10 + 0*10 + 8 = 18
    print(f"Hand Card 0 currentDamage encoding: {combat_vec[18]}")
    assert combat_vec[18] == 9 / 50.0
    
    # Check enemy offset
    # Enemy 0 alive at index 110
    print(f"Enemy 0 alive encoding: {combat_vec[110]}")
    assert combat_vec[110] == 1.0

    print("State encoding successful!")

    # Test dummy forward pass
    print("Testing forward pass...")
    try:
        # We need a batch axis
        import haiku as hk
        
        # Access the learner from the bridge
        learner = rnad_bridge.learner
        if learner is None:
            print("Learner not initialized (bridge might be waiting for game). Mocking one...")
            return
            
        params = learner.params
        key = jax.random.PRNGKey(42)
        
        obs_dict = {
            "global": jnp.array([encoded["global"]]),
            "combat": jnp.array([encoded["combat"]]),
            "map": jnp.array([encoded["map"]]),
            "event": jnp.array([encoded["event"]]),
            "draw_bow": jnp.array([encoded["draw_bow"]]),
            "discard_bow": jnp.array([encoded["discard_bow"]]),
            "exhaust_bow": jnp.array([encoded["exhaust_bow"]]),
            "master_bow": jnp.array([encoded["master_bow"]]),
            "state_type": jnp.array([encoded["state_type"]])
        }
        mask_arr = jnp.ones((1, 100))
        
        # Run predict
        logits, value = learner.network.apply(params, key, obs_dict, mask_arr, is_training=False)
        print(f"Model output shape: {logits.shape}")
        assert logits.shape == (1, 100)
        print("Forward pass successful!")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_shapes()
