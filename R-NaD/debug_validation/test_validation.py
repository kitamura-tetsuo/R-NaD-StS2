import os
import json
import sys
import numpy as np

# Add the project directory to sys.path
sys.path.append("/home/ubuntu/src/R-NaD-StS2/R-NaD")

from rnad_bridge import encode_state, validate_encoding, log

def test_validation():
    state = {
        "type": "combat",
        "floor": 1,
        "gold": 99,
        "player": {
            "hp": 75,
            "maxHp": 80,
            "block": 5,
            "energy": 2,
            "stars": 0,
            "drawPile": [{"id": "Strike_R"}],
            "discardPile": [{"id": "Defend_R"}],
            "exhaustPile": [],
            "masterDeck": [{"id": "Strike_R"}, {"id": "Defend_R"}],
            "powers": [{"id": "Strength", "amount": 3}]
        },
        "hand": [
            {"id": "Strike_R", "isPlayable": True, "targetType": "SingleEnemy", "cost": 1, "baseDamage": 6, "baseBlock": 0, "magicNumber": 0, "upgraded": False, "currentDamage": 9, "currentBlock": 0}
        ],
        "enemies": [
            {
                "id": "Nibbit", "hp": 15, "maxHp": 20, "block": 0, "isMinion": False, 
                "intents": [{"type": "Attack", "damage": 6, "repeats": 1, "count": 1}],
                "powers": []
            }
        ],
        "potions": [{"id": "Fire Potion"}],
        "relics": [{"id": "Burning Blood"}]
    }

    print(f"Testing validation for state type: {state.get('type')}")
    
    # 1. Encode
    print("Encoding state...")
    encoded = encode_state(state)
    
    # 2. Validate
    print("Running validate_encoding...")
    success = validate_encoding(state, encoded)
    
    if success:
        print("SUCCESS: Validation passed!")
    else:
        print("FAILURE: Validation failed with discrepancies.")

if __name__ == "__main__":
    test_validation()
