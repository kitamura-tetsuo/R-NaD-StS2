import pickle
import numpy as np
import os
import glob
import time

try:
    import grain.python as grain
except ImportError:
    grain = None

ARRAY_RECORD_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD/trajectories/array_record"

def check_grain_dataset(paths):
    print(f"Checking datasets: {paths}...")
    dataset = grain.MapDataset.source(grain.ArrayRecordDataSource(paths))
    dataset = dataset.map(lambda x: pickle.loads(x))
    
    count = 0
    for i, segment in enumerate(dataset):
        steps = segment.get('steps', [])
        for j, step in enumerate(steps):
            rew = step.get('rew', 0.0)
            lp = step.get('log_prob', 0.0)
            mask = step.get('mask', np.ones(100))
            
            if np.isnan(rew) or np.isinf(rew):
                print(f"  [NAN/INF REWARD] segment: {i}, step: {j}, rew: {rew}")
                print(f"  Source: {segment.get('source')}, range: {segment.get('range')}")
            
            if np.isnan(lp) or np.isinf(lp):
                print(f"  [NAN/INF LOGPROB] segment: {i}, step: {j}, lp: {lp}")
                print(f"  Source: {segment.get('source')}, range: {segment.get('range')}")

            if abs(rew) > 1000:
                print(f"  [EXTREME REWARD] segment: {i}, step: {j}, rew: {rew}")
                print(f"  Source: {segment.get('source')}, range: {segment.get('range')}")
            
            if np.sum(mask) == 0:
                print(f"  [EMPTY MASK] segment: {i}, step: {j}")
                print(f"  Source: {segment.get('source')}, range: {segment.get('range')}")
                
        count += 1
    print(f"Checked {count} segments.")

def check_raw_jsons():
    TRAJECTORY_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD/trajectories"
    files = glob.glob(os.path.join(TRAJECTORY_DIR, "traj_*.json")) + \
            glob.glob(os.path.join(TRAJECTORY_DIR, "bundled_*.json"))
    
    for filepath in files:
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            traj_list = data if isinstance(data, list) else [data]
            for traj in traj_list:
                steps = traj.get('steps', [])
                for idx, step in enumerate(steps):
                    reward = step.get('reward', 0.0)
                    mask = step.get('mask', [1]*100)
                    if np.isnan(reward) or np.isinf(reward):
                        print(f"  [NAN/INF REWARD] file: {filepath}, step: {idx}, rew: {reward}")
                    if abs(reward) > 1000:
                        print(f"  [EXTREME REWARD] file: {filepath}, step: {idx}, rew: {reward}")
                    if sum(mask) == 0:
                        print(f"  [EMPTY MASK] file: {filepath}, step: {idx}")
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    machine_record = os.path.join(ARRAY_RECORD_DIR, "machine_data.array_record")
    human_record = os.path.join(ARRAY_RECORD_DIR, "human_data.array_record")
    
    paths = []
    if os.path.exists(machine_record): paths.append(machine_record)
    if os.path.exists(human_record): paths.append(human_record)
    
    if not paths or grain is None:
        print("Falling back to raw JSONs...")
        check_raw_jsons()
    else:
        check_grain_dataset(paths)
