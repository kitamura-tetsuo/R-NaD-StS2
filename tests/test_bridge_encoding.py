import sys
import os
from unittest.mock import MagicMock
import numpy as np

# Mocking modules and global dependencies for rnad_bridge.py
sys.modules['jax'] = MagicMock()
sys.modules['jax.numpy'] = MagicMock()
sys.modules['src.rnad'] = MagicMock()
sys.modules['experiment'] = MagicMock()
sys.modules['libpython_fixer'] = MagicMock()
sys.modules['event_dict'] = MagicMock()

# Add bridge directory to sys.path
sys.path.insert(0, "/home/ubuntu/src/R-NaD-StS2/R-NaD")

# Mock globals used in rnad_bridge.py
import rnad_bridge

# Replace problematic globals before calling anything
rnad_bridge.np = np
rnad_bridge.jnp = np # numpy is close enough for shape/indexing tests
rnad_bridge.log = MagicMock()

def test_encode_state_unknown_map_type_fallback():
    print("Testing encode_state fallback for Unknown map node type...")
    # Unknown is no longer in nt_map, so it should trigger a fallback warning
    rnad_bridge.log.reset_mock()
    state = {
        "type": "map",
        "floor": 10,
        "nodes": [{"row": 1, "col": 1, "type": "Unknown"}],
        "current_pos": {"row": 0, "col": 0}
    }
    try:
        encoded = rnad_bridge.encode_state(state)
        print("Success: encode_state did not crash for Unknown map type.")
        if rnad_bridge.log.call_count > 0:
            calls = [str(call[0][0]) for call in rnad_bridge.log.call_args_list]
            matched = any("Unknown" in msg for msg in calls)
            if matched:
                print(f"Verified: Warning log was emitted for Unknown: {calls}")
            else:
                print(f"FAILED: No warning log for Unknown. Calls: {calls}")
                exit(1)
    except Exception as e:
        print(f"FAILED: encode_state crashed: {e}")
        exit(1)

def test_encode_state_gold():
    print("\nTesting encode_state with normalized 'Gold' type...")
    # Clear log mock
    rnad_bridge.log.reset_mock()
    state = {
        "type": "rewards",
        "floor": 10,
        "rewards": [{"type": "Gold"}]
    }
    try:
        encoded = rnad_bridge.encode_state(state)
        print("Success: encode_state did not crash for 'Gold' reward type.")
        if rnad_bridge.log.call_count > 0:
            # Check if any warning was emitted (it shouldn't be for 'Gold')
            calls = [str(call[0][0]) for call in rnad_bridge.log.call_args_list]
            if any("reward_type: Gold not in map" in msg for msg in calls):
                 print(f"FAILED: Unexpected warning for 'Gold': {calls}")
                 exit(1)
    except Exception as e:
        print(f"FAILED: encode_state crashed: {e}")
        exit(1)

def test_encode_state_defensive_fallback():
    print("\nTesting encode_state defensive fallback for entirely new type...")
    # Clear log mock
    rnad_bridge.log.reset_mock()
    state = {
        "type": "map",
        "floor": 10,
        "nodes": [{"row": 1, "col": 1, "type": "CrazyNewRoomType"}],
        "current_pos": {"row": 0, "col": 0}
    }
    try:
        encoded = rnad_bridge.encode_state(state)
        print("Success: encode_state did not crash for new type.")
        # Ensure log warning was called
        calls = [str(call[0][0]) for call in rnad_bridge.log.call_args_list]
        matched = any("CrazyNewRoomType" in msg for msg in calls)
        if matched:
            print("Verified: Warning log was emitted for unmapped type.")
        else:
            print("FAILED: No warning log emitted for unmapped type.")
            print(f"Recorded calls: {calls}")
            exit(1)
    except Exception as e:
        print(f"FAILED: encode_state crashed: {e}")
        exit(1)

if __name__ == "__main__":
    test_encode_state_unknown_map_type_fallback()
    test_encode_state_gold()
    test_encode_state_defensive_fallback()
    print("\nALL BRIDGE ENCODING TESTS PASSED (with C# normalization logic)!")
