import os
import sys
import argparse
import json
import numpy as np
import jax
import jax.numpy as jnp
import pickle
import traceback

# Set environment variable to skip default R-NaD initialization
os.environ["SKIP_RNAD_INIT"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"  # Debugging on CPU is usually enough
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Force CPU

# Add the R-NaD directory to sys.path
R_NAD_DIR = os.path.dirname(os.path.abspath(__file__))
if R_NAD_DIR not in sys.path:
    sys.path.insert(0, R_NAD_DIR)

import rnad_bridge

def validate_segment(source_file, start_idx, end_idx, checkpoint=None, unroll_length=None):
    print(f"--- Validating Data Source: {source_file} (Target Range: {start_idx}-{end_idx}) ---")
    
    # Standardize path
    if not os.path.isabs(source_file):
        if not os.path.exists(source_file):
            source_file = os.path.join(R_NAD_DIR, source_file)
    
    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        return

    # Load model
    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
    rnad_bridge.load_model(checkpoint_path=checkpoint)
    
    worker = rnad_bridge.training_worker
    if worker is None:
        print("Error: TrainingWorker failed to initialize.")
        return

    if unroll_length:
        print(f"Setting unroll_length to: {unroll_length}")
        worker.config = worker.config._replace(unroll_length=unroll_length)

    # Extract the segment
    is_human = "human_play" in source_file
    print(f"Is Human Data: {is_human}")
    
    matching_segments = []
    all_segments_count = 0
    try:
        segments_gen = worker._get_segments_from_raw_files([source_file], is_human)
        for seg in segments_gen:
            all_segments_count += 1
            if seg["range"] == (start_idx, end_idx):
                matching_segments.append(seg)
    except Exception as e:
        print(f"Error loading segments: {e}")
        traceback.print_exc()
        return

    if not matching_segments:
        print(f"Error: Could not find any segment with range ({start_idx}, {end_idx}) in {source_file}")
        print(f"Found {all_segments_count} segments in total.")
        return

    print(f"Found {len(matching_segments)} matching segments. Validating each...")
    
    for seg_idx, found_segment in enumerate(matching_segments):
        print(f"\n--- Validating Segment {seg_idx+1}/{len(matching_segments)} ---")
        
        # Perform update with single segment (batch size 1)
        try:
            # Save original batch size
            orig_batch_size = worker.config.batch_size
            worker.config = worker.config._replace(batch_size=1)
            
            # Check for NaNs in observations
            for step_idx, step in enumerate(found_segment["steps"]):
                obs = step["obs"]
                for k, v in obs.items():
                    if isinstance(v, (np.ndarray, jnp.ndarray)):
                        if np.isnan(v).any():
                            print(f"  CRITICAL: NaN found in observation '{k}' at step {step_idx}")
            
            worker.perform_update([found_segment], increment_step=False, reset_updating=True)
            
            # Inspect the segment data for potential issues
            steps = found_segment["steps"]
            rewards = [s["rew"] for s in steps]
            log_probs = [s["log_prob"] for s in steps]
            
            print(f"  Length: {len(steps)}")
            print(f"  Rewards: min={min(rewards):.4f}, max={max(rewards):.4f}, mean={sum(rewards)/len(rewards):.4f}")
            print(f"  Log Probs: min={min(log_probs):.4f}, max={max(log_probs):.4f}, mean={sum(log_probs)/len(log_probs):.4f}")
            
            for i, s in enumerate(steps):
                if np.isnan(s["rew"]) or np.isinf(s["rew"]):
                    print(f"  Step {i}: INVALID REWARD {s['rew']}")
                if np.isnan(s["log_prob"]) or np.isinf(s["log_prob"]):
                    print(f"  Step {i}: INVALID LOG_PROB {s['log_prob']}")

        except Exception as e:
            print(f"  Error during update of segment {seg_idx}: {e}")
            traceback.print_exc()
        finally:
            worker.config = worker.config._replace(batch_size=orig_batch_size)

def main():
    parser = argparse.ArgumentParser(description="Validate specific trajectory data for NaNs")
    parser.add_argument("--source", type=str, required=True, help="Path to the source file")
    parser.add_argument("--start", type=int, required=True, help="Start index of the range")
    parser.add_argument("--end", type=int, required=True, help="End index of the range")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint to load")
    parser.add_argument("--unroll_length", type=int, default=48, help="Unroll length used during training")
    
    args = parser.parse_args()
    
    validate_segment(args.source, args.start, args.end, args.checkpoint, args.unroll_length)

if __name__ == "__main__":
    main()
