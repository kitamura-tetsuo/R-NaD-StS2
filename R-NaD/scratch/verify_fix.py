import sys
import os
import json
import numpy as np

# Add the project directory to path
sys.path.append("/home/ubuntu/src/R-NaD-StS2/R-NaD")

import rnad_bridge

# Ensure numpy and other dependencies are loaded in the bridge
rnad_bridge.do_deferred_imports()

def check_human_file_rewards(filepath):
    print(f"Checking rewards for {filepath}...")
    rnad_bridge.reward_tracker.reset_for_new_run()
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    last_was_terminal = False
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        step_data = json.loads(line)
        state = step_data.get("state")
        if not state or step_data.get("action_id") is None: continue
        
        # Reset RewardTracker if the previous step was terminal or if floor dropped (new run)
        current_floor = state.get("floor", 0)
        if last_was_terminal or (current_floor < rnad_bridge.reward_tracker.last_processed_floor and current_floor <= 1):
            rnad_bridge.reward_tracker.reset_for_new_run()
        
        action_idx = int(step_data["action_id"])
        state_type = state.get("type", "unknown")
        
        # This calls the newly modified compute_intermediate_reward
        reward = rnad_bridge.compute_reward(state, state_type) + rnad_bridge.compute_intermediate_reward(state, state_type, action_idx)
        
        if abs(reward) > 10.01: # Allow slight floating point tolerance
            print(f"  [EXTREME REWARD] line: {i}, rew: {reward}")
            
        last_was_terminal = (state_type == "game_over")

if __name__ == "__main__":
    # Check the known problematic files
    check_human_file_rewards("/mnt/nas/StS2/replay/human_play_BS932RQLC7.jsonl")
    check_human_file_rewards("/mnt/nas/StS2/replay/human_play_C4ZKB0RA5A.jsonl")
    print("Done checking.")
