import numpy as np
import sys
import os

# Mocking modules and environment for testing encode_state
import jax.numpy as jnp
from unittest.mock import MagicMock

# Add R-NaD to path
BRIDGE_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD"
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)

# Mock some parts of rnad_bridge to avoid loading the whole thing (which might fail due to venv/ctypes)
import rnad_bridge

def test_event_encoding():
    # Mock a state for ShiningLight event
    state = {
        "type": "event",
        "floor": 10,
        "id": "ShiningLight",
        "options": [
            {"index": 0, "title": "Enter", "is_locked": False},
            {"index": 1, "title": "Leave", "is_locked": False}
        ],
        "player": {
            "hp": 80,
            "maxHp": 100,
            "gold": 100
        }
    }
    
    encoded = rnad_bridge.encode_state(state)
    event_vec = encoded["event"]
    
    print(f"Event Vector Shape: {event_vec.shape}")
    assert event_vec.shape == (128,)
    
    # Check option 0 locked status (index 0)
    print(f"Option 0 locked status (1.0 = unlocked): {event_vec[0]}")
    assert event_vec[0] == 1.0
    
    # Check rich features for ShiningLight Option 0 (offset 20)
    # expected: {"hp_loss_pct": 0.2, "card_upgrade_count": 2}
    # params mapping in get_event_features:
    # 0: hp_loss_pct
    # 5: card_upgrade_count / 5.0
    rich_base = 20
    hp_loss_pct = event_vec[rich_base]
    upgrade_count_norm = event_vec[rich_base + 5]
    
    print(f"Rich Feature - HP Loss Pct: {hp_loss_pct}")
    print(f"Rich Feature - Card Upgrade Count (Normalized): {upgrade_count_norm}")
    
    assert hp_loss_pct == 0.2
    assert upgrade_count_norm == 2.0 / 5.0
    
    # Check differentiation flag (index 90 for size 128)
    # Permanent grid = 1.0, Hand = -1.0, Event = 0.0 (default)
    print(f"Differentiation Flag: {event_vec[90]}")
    assert event_vec[90] == 0.0
    
    print("Test encoding for permanently grid selection")
    grid_state = {
        "type": "grid_selection",
        "cards": [{"id": "Strike_R", "upgraded": False, "cost": 1}]
    }
    encoded_grid = rnad_bridge.encode_state(grid_state)
    print(f"Encoded Grid State Type Index: {encoded_grid['state_type']}")
    print(f"Encoded Grid Differentiation Flag (index 90): {encoded_grid['event'][90]}")
    assert encoded_grid["event"][90] == 1.0
    
    print("SUCCESS: Event encoding verification passed.")

if __name__ == "__main__":
    try:
        test_event_encoding()
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
