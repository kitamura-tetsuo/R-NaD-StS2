import sys
import os
import json
from unittest.mock import MagicMock

# Add current directory to sys.path to import rnad_bridge
sys.path.append(os.path.join(os.getcwd(), "R-NaD"))

# Mocking the predict_action function and its attributes
def mock_predict_action(state_json):
    pass

mock_predict_action.last_processed_floor = -1
mock_predict_action.last_player_hp = 0
mock_predict_action.last_total_enemy_hp = 0

# Import the module after setting up sys.path
import rnad_bridge
# Replace the actual predict_action with our mock in the module
# We need to ensure that the rnad_bridge module uses our mock for side effects
# Since rnad_bridge.compute_intermediate_reward uses 'predict_action' (the function object)
# and its attributes, we should set them on the function that and compute_intermediate_reward expects.

rnad_bridge.predict_action = mock_predict_action

def test_rewards():
    # Finalize reward: Victory or Defeat
    print("Testing compute_reward (Terminal)...")
    state_victory = {"victory": True}
    reward = rnad_bridge.compute_reward(state_victory, "game_over")
    print(f"  Victory reward: {reward} (Expected: 5.0)")
    assert reward == 5.0

    state_defeat = {"victory": False}
    reward = rnad_bridge.compute_reward(state_defeat, "game_over")
    print(f"  Defeat reward: {reward} (Expected: -1.0)")
    assert reward == -1.0

    print("\nTesting compute_intermediate_reward (Floor Progression)...")
    # Initial floor (0 -> 1)
    state_f1 = {"floor": 1, "player": {"hp": 72}, "enemies": []}
    reward = rnad_bridge.compute_intermediate_reward(state_f1, "map", None)
    print(f"  Floor 1 initialization reward: {reward} (Expected: 0.0)")
    assert reward == 0.0
    assert rnad_bridge.predict_action.last_processed_floor == 1
    assert rnad_bridge.predict_action.last_player_hp == 72
    assert rnad_bridge.predict_action.last_total_enemy_hp == 0

    # Floor progression (1 -> 2)
    state_f2 = {"floor": 2, "player": {"hp": 70}, "enemies": []}
    reward = rnad_bridge.compute_intermediate_reward(state_f2, "map", None)
    print(f"  Floor 2 progression reward: {reward} (Expected: 0.1)")
    assert abs(reward - 0.1) < 1e-6
    assert rnad_bridge.predict_action.last_processed_floor == 2
    assert rnad_bridge.predict_action.last_player_hp == 70

    print("\nTesting compute_intermediate_reward (Combat Deltas)...")
    # Manually initialize trackers for combat test to avoid start-of-combat jump
    rnad_bridge.predict_action.last_total_enemy_hp = 80
    rnad_bridge.predict_action.last_player_hp = 70
    
    # Deal damage
    state_c2 = {"floor": 2, "type": "combat", "player": {"hp": 70}, "enemies": [{"hp": 40}, {"hp": 30}]}
    reward = rnad_bridge.compute_intermediate_reward(state_c2, "combat", None)
    # damage_dealt = 80 - 70 = 10. reward = 10 * 0.002 = 0.02
    print(f"  Damage dealt (10) reward: {reward} (Expected: 0.02)")
    assert abs(reward - 0.02) < 1e-6
    assert rnad_bridge.predict_action.last_total_enemy_hp == 70

    # Take damage
    state_c3 = {"floor": 2, "type": "combat", "player": {"hp": 60}, "enemies": [{"hp": 40}, {"hp": 30}]}
    reward = rnad_bridge.compute_intermediate_reward(state_c3, "combat", None)
    # damage_taken = 70 - 60 = 10. reward = 10 * -0.005 = -0.05
    print(f"  Damage taken (10) reward: {reward} (Expected: -0.05)")
    assert abs(reward - (-0.05)) < 1e-6
    assert rnad_bridge.predict_action.last_player_hp == 60

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_rewards()
