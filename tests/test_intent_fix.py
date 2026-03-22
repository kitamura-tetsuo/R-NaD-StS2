import os
import sys
import json
import numpy as np

# Add the R-NaD directory to sys.path
BRIDGE_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD"
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)

# Mock some environment variables to avoid GPU/Server issues
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

try:
    from rnad_bridge import encode_state, do_deferred_imports
    print("Successfully imported encode_state from rnad_bridge")
except ImportError as e:
    print(f"Failed to import rnad_bridge: {e}")
    sys.exit(1)

def test_debuff_strong():
    # Mock state with DebuffStrong intent
    state = {
        "type": "combat",
        "floor": 10,
        "gold": 100,
        "player": {
            "hp": 50,
            "maxHp": 80,
            "block": 0,
            "energy": 3,
            "powers": [],
            "drawPile": [],
            "discardPile": [],
            "exhaustPile": [],
            "masterDeck": []
        },
        "hand": [],
        "enemies": [
            {
                "id": "Cultist",
                "hp": 50,
                "maxHp": 50,
                "block": 0,
                "intents": [
                    {"type": "DebuffStrong", "damage": 0, "repeats": 1}
                ],
                "powers": []
            }
        ],
        "potions": [],
        "relics": ["Burning Blood"]
    }
    
    print("Testing encode_state with DebuffStrong intent...")
    try:
        # We need to ensure deferred imports are done so np is available
        do_deferred_imports()
        
        obs = encode_state(state)
        print("Successfully encoded state with DebuffStrong!")
        
        # Verify the value in combat_vec
        # Enemy 0 starts at index 110 in combat_vec
        # Intent 0 is at index 110 + 4 = 114
        combat_vec = obs["combat"]
        intent_val = combat_vec[114]
        print(f"Intent value in combat_vec[114]: {intent_val}")
        
        # DebuffStrong (6) / 10.0 = 0.6
        assert abs(intent_val - 0.6) < 1e-6, f"Expected 0.6, got {intent_val}"
        print("Assertion passed: DebuffStrong correctly mapped to 0.6")
        
    except AssertionError as e:
        print(f"Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def test_attack_defense():
    # Mock state with AttackDefense intent to verify the fix for existing intents
    state = {
        "type": "combat",
        "floor": 10,
        "player": {"hp": 50, "maxHp": 80, "drawPile": [], "discardPile": [], "exhaustPile": [], "masterDeck": []},
        "enemies": [
            {
                "id": "Slaver",
                "hp": 50,
                "maxHp": 50,
                "intents": [
                    {"type": "AttackDefense", "damage": 10, "repeats": 1}
                ]
            }
        ]
    }
    
    print("Testing encode_state with AttackDefense intent...")
    try:
        obs = encode_state(state)
        intent_val = obs["combat"][114]
        print(f"Intent value in combat_vec[114]: {intent_val}")
        # AttackDefense (3) / 10.0 = 0.3
        assert abs(intent_val - 0.3) < 1e-6, f"Expected 0.3, got {intent_val}"
        print("Assertion passed: AttackDefense correctly mapped to 0.3")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_debuff_strong()
    test_attack_defense()
    print("All tests passed!")
