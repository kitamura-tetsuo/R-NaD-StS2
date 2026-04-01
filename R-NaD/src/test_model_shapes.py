import os
import sys
import numpy as np

# Add the R-NaD directory to the path to import from src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

# Mock some environment variables and global values that bridge might expect
os.environ["ST_DEBUG"] = "1"
os.environ["SKIP_RNAD_INIT"] = "1"

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
                "id": "Cultist",
                "hp": 40,
                "maxHp": 40,
                "block": 0,
                "intents": [{"type": "Attack", "damage": 6, "repeats": 1}]
            }
        ],
        "potions": [{"id": "empty"}, {"id": "empty"}],
        "type": "combat"
    }

    # Add a global state for the old version if it expects it
    dummy_state["relics"] = []
    dummy_state["floorNum"] = 1
    dummy_state["money"] = 100
    dummy_state["hp"] = 80
    dummy_state["maxHp"] = 80
    dummy_state["energy"] = 3
    dummy_state["hand"] = dummy_state["hand"] # Move hand if needed
    dummy_state["discardPile"] = []
    dummy_state["drawPile"] = []
    dummy_state["exhaustPile"] = []
    dummy_state["powers"] = []

    print("Encoding state...")
    encoded = rnad_bridge.encode_state(dummy_state)
    
    combat_vec = encoded["combat"]
    print(f"Combat vector shape: {combat_vec.shape}")
    assert combat_vec.shape == (384,)
    
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
        
        # Access the learner from the bridge or create one
        learner = rnad_bridge.learner
        if learner is None:
            print("Learner not initialized. Creating a fresh one for testing...")
            from src.rnad import RNaDConfig, RNaDLearner
            config = RNaDConfig(
                hidden_size=256, num_heads=4, num_blocks=1,
                card_vocab_size=rnad_bridge.VOCAB_SIZE,
                monster_vocab_size=rnad_bridge.MONSTER_VOCAB_SIZE,
                relic_vocab_size=rnad_bridge.RELIC_VOCAB_SIZE,
                power_vocab_size=rnad_bridge.POWER_VOCAB_SIZE
            )
            learner = RNaDLearner(0, num_actions=100, config=config)
            learner.init(jax.random.PRNGKey(0))
            
        params = learner.params
        key = jax.random.PRNGKey(42)
        
        # Test all 6 heads
        for ht_idx in range(6):
            print(f"Testing head_type: {ht_idx}")
            obs_dict = {
                "global": jnp.array([[encoded["global"]]]),
                "combat": jnp.array([[encoded["combat"]]]),
                "relic_ids": jnp.array([[encoded["relic_ids"]]]),
                "map": jnp.array([[encoded["map"]]]),
                "event": jnp.array([[encoded["event"]]]),
                "draw_bow": jnp.array([[encoded["draw_bow"]]]),
                "discard_bow": jnp.array([[encoded["discard_bow"]]]),
                "exhaust_bow": jnp.array([[encoded["exhaust_bow"]]]),
                "master_bow": jnp.array([[encoded["master_bow"]]]),
                "state_type": jnp.array([[encoded["state_type"]]]), # (T=1, B=1)
                "head_type": jnp.array([[ht_idx]])  # (T=1, B=1)
            }
            mask_arr = jnp.ones((1, 1, 100)) # (T=1, B=1, num_actions)
            
            # Run predict
            logits, value = learner.network.apply(params, key, obs_dict, mask_arr, is_training=False)
            print(f"  Model output shape: logits={logits.shape}, value={value.shape}")
            assert logits.shape == (1, 1, 100)
            assert value.shape == (1, 1)
        
        print("Forward pass successful for all heads!")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_shapes()
