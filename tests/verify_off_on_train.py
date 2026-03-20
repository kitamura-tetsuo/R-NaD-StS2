import requests
import json
import time
import os
import glob

BRIDGE_URL = "http://127.0.0.1:8081"
TRAJ_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD/trajectories"

def wait_for_server():
    print("Waiting for bridge server and TrainingWorker...")
    for _ in range(60):
        try:
            resp = requests.get(f"{BRIDGE_URL}/status", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if "step_count" in data:
                    print("Server and worker seem ready!")
                    return True
        except:
            pass
        time.sleep(1)
    return False

def create_mock_trajectory():
    print(f"Creating mock trajectory in {TRAJ_DIR}...")
    if not os.path.exists(TRAJ_DIR):
        os.makedirs(TRAJ_DIR, exist_ok=True)
    
    # Cleanup previous mock trajs
    for f in glob.glob(os.path.join(TRAJ_DIR, "traj_MOCK_*.json")):
        os.remove(f)
        
    timestamp = "MOCK"
    episode_idx = "1"
    
    # Create 128 steps to satisfy unroll_length, saved in a single file to match the episode-based flush
    filename = f"traj_{timestamp}_{episode_idx}.json"
    filepath = os.path.join(TRAJ_DIR, filename)
    
    steps = []
    for i in range(128):
        state = {
            "screen_type": "combat", 
            "floor": 1, 
            "player": {"hp": 80, "max_hp": 80},
            "enemies": [{"hp": 10, "max_hp": 10}]
        }
        
        step_data = {
            "state_json": json.dumps(state),
            "action_idx": 5,
            "probs": [0.0] * 400,
            "mask": [False] * 400,
            "reward": 0.0
        }
        step_data["probs"][5] = 1.0
        step_data["mask"][5] = True
        steps.append(step_data)
        
    with open(filepath, "w") as f:
        json.dump({
            "semantic_map": {"MOCK": "MAP"},
            "steps": steps
        }, f)
            
    print(f"Created mock trajectory episode at {filepath}")
            
    print(f"Created 128 mock trajectory steps at {TRAJ_DIR}/traj_{timestamp}_{episode_idx}_*.json")
    return TRAJ_DIR

def test_flow():
    # Cleanup old trajectories
    if os.path.exists(TRAJ_DIR):
        for f in glob.glob(os.path.join(TRAJ_DIR, "traj_*.json")):
            os.remove(f)
            
    create_mock_trajectory()
    
    if not wait_for_server():
        print("Error: Bridge server or TrainingWorker not ready.")
        return

    # Trigger offline training
    print("Triggering offline training...")
    try:
        resp = requests.get(f"{BRIDGE_URL}/offline_train", timeout=5)
        print(f"Offline train response: {resp.status_code} - {resp.text}")
        if resp.status_code == 200:
            print(f"Offline training started: {resp.json()}")
        else:
            print("Offline training failed to start.")
            return
    except Exception as e:
        print(f"Error calling /offline_train: {e}")
        return
    
    # Wait for offline training to finish
    print("Waiting for offline training to complete...")
    initial_step = requests.get(f"{BRIDGE_URL}/status").json().get("step_count", 0)
    print(f"Initial step count: {initial_step}")
    
    for _ in range(60): # Increase timeout for CPU
        resp = requests.get(f"{BRIDGE_URL}/status")
        data = resp.json()
        current_step = data.get("step_count", 0)
        
        if data.get("worker_error"):
            print(f"Worker encountered error: {data.get('worker_error')}")
            break
            
        if current_step > initial_step:
            print(f"Offline training successful! Step count increased to {current_step}")
            break
            
        print(f"Still at step {current_step}. Worker is_updating={data.get('is_updating')}")
        time.sleep(2)
    
    # Verify semantic map is inside the trajectory JSON
    mock_traj = os.path.join(TRAJ_DIR, "traj_MOCK_1.json")
    if not os.path.exists(mock_traj):
        print(f"Error: {mock_traj} missing.")
        return
    
    with open(mock_traj, "r") as f:
        data = json.load(f)
        if "semantic_map" in data:
            print("Success: Semantic map found inside trajectory JSON.")
            print(f"Map keys: {list(data['semantic_map'].keys())}")
        else:
            print("Error: Semantic map missing from trajectory JSON.")

if __name__ == "__main__":
    test_flow()
