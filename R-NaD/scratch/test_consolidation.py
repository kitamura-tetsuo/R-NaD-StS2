import os
import json
import shutil
import datetime

# Mock TRAJECTORY_DIR
TEST_DIR = "test_trajectories"
if os.path.exists(TEST_DIR):
    shutil.rmtree(TEST_DIR)
os.makedirs(TEST_DIR)

def log(msg):
    print(msg)

# Create mock trajectory files
for i in range(1005):
    filepath = os.path.join(TEST_DIR, f"traj_{i}.json")
    with open(filepath, "w") as f:
        json.dump({"steps": [{"id": i}]}, f)

print(f"Created {len(os.listdir(TEST_DIR))} mock files.")

def consolidate_trajectories(trajectory_dir):
    """Bundle groups of 1000 individual trajectory files into single files."""
    log(f"Checking for trajectory consolidation in {trajectory_dir}...")
    import glob
    
    all_files = glob.glob(os.path.join(trajectory_dir, "traj_*.json"))
    files = sorted([f for f in all_files if os.path.basename(f).startswith("traj_")])
    
    if len(files) < 1000:
        log(f"Not enough trajectories to consolidate ({len(files)} < 1000).")
        return

    batch_size = 1000
    num_batches = len(files) // batch_size
    log(f"Consolidating {num_batches * batch_size} files into {num_batches} bundles...")
    
    for i in range(num_batches):
        batch = files[i * batch_size : (i + 1) * batch_size]
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        bundle_filename = f"bundled_{timestamp}_{i}.json"
        bundle_path = os.path.join(trajectory_dir, bundle_filename)
        
        bundled_data = []
        for filepath in batch:
            with open(filepath, "r") as f:
                bundled_data.append(json.load(f))
        
        if bundled_data:
            with open(bundle_path, "w") as f:
                json.dump(bundled_data, f)
            log(f"Successfully bundled {len(bundled_data)} trajectories into {bundle_path}")
            
            for filepath in batch:
                os.remove(filepath)
                
    log("Trajectory consolidation complete.")

# Run consolidation
consolidate_trajectories(TEST_DIR)

# Check results
remaining_files = os.listdir(TEST_DIR)
print(f"Remaining files: {len(remaining_files)}")
print(f"Files: {remaining_files}")

# Verify bundled file content
bundled_files = [f for f in remaining_files if f.startswith("bundled_")]
if bundled_files:
    with open(os.path.join(TEST_DIR, bundled_files[0]), "r") as f:
        data = json.load(f)
        print(f"Bundled data size: {len(data)}")
        print(f"First element: {data[0]}")

# Clean up
shutil.rmtree(TEST_DIR)
