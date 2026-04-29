import os
import sys
import argparse
import glob
import time
import traceback

# Set environment variable to skip default R-NaD initialization
os.environ["SKIP_RNAD_INIT"] = "1"

# os.environ["JAX_PLATFORMS"] = "cpu"

# Add the R-NaD directory to sys.path
R_NAD_DIR = os.path.dirname(os.path.abspath(__file__))
if R_NAD_DIR not in sys.path:
    sys.path.insert(0, R_NAD_DIR)

def main():
    parser = argparse.ArgumentParser(description="Offline Training for R-NaD StS2")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint .pkl to resume from")
    parser.add_argument("--epochs", type=int, default=100000, help="Number of passes through all found trajectories")
    parser.add_argument("--save_interval", type=int, default=1, help="Frequency (in epochs) to save checkpoints and log to MLflow")
    parser.add_argument("--data_dir", type=str, help="Directory containing human play data (.jsonl)")
    parser.add_argument("--trajectory_dir", type=str, help="Directory containing machine trajectories (.json)")
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to search for/save checkpoints")
    parser.add_argument("--jax_platform", type=str, choices=["cpu", "gpu", "tpu"], help="JAX platform to use")
    args = parser.parse_args()

    # Apply environment overrides based on arguments
    if args.data_dir:
        os.environ["RNAD_REPLAY_DIR"] = os.path.abspath(args.data_dir)
        print(f"Setting RNAD_REPLAY_DIR to: {os.environ['RNAD_REPLAY_DIR']}")
    
    if args.trajectory_dir:
        os.environ["RNAD_TRAJECTORY_DIR"] = os.path.abspath(args.trajectory_dir)
        print(f"Setting RNAD_TRAJECTORY_DIR to: {os.environ['RNAD_TRAJECTORY_DIR']}")
    
    if args.jax_platform:
        os.environ["JAX_PLATFORMS"] = args.jax_platform
        print(f"Setting JAX_PLATFORMS to: {os.environ['JAX_PLATFORMS']}")
    elif "JAX_PLATFORMS" not in os.environ:
        # Default to gpu if not set, instead of hardcoded cpu in the original script
        os.environ["JAX_PLATFORMS"] = "gpu"
        print(f"JAX_PLATFORMS not set, defaulting to: {os.environ['JAX_PLATFORMS']}")

    # Standard imports after setting environment variables
    import rnad_bridge
    from download_human_data import download_human_data

    print(f"--- Starting Offline Training (Epochs: {args.epochs}) ---")
    
    # Auto-detect latest checkpoint if not specified
    if args.checkpoint is None:
        checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else os.path.join(R_NAD_DIR, "checkpoints")
        if os.path.exists(checkpoint_dir):
            # Recursively search for checkpoint_*.pkl
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "**", "checkpoint_*.pkl"), recursive=True)
            if checkpoints:
                # Find the latest by modification time
                args.checkpoint = max(checkpoints, key=os.path.getmtime)
                print(f"Auto-detected latest checkpoint: {args.checkpoint}")
            else:
                print(f"No checkpoints found in '{checkpoint_dir}' directory.")
        else:
            print(f"Checkpoint directory not found: {checkpoint_dir}")

    try:
        # 1. Load the model and initialize the TrainingWorker
        # Note: load_model internally handles configuration and ExperimentManager setup
        rnad_bridge.load_model(checkpoint_path=args.checkpoint)
        
        # 2. Wait for initialization if necessary (usually synchronous in load_model)
        if rnad_bridge.training_worker is None:
            print("Error: TrainingWorker failed to initialize.")
            return

        # 3. Trigger offline training
        # Note: TrainingWorker.perform_offline_training() loads trajectories/human replays
        # and runs the update loop.
        last_save_time = time.time()
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            
            # Download latest human play data from Discord before each epoch
            try:
                download_human_data(replay_dir=args.data_dir)
            except Exception as e:
                print(f"Warning: Failed to download human data: {e}")
            
            # Save checkpoint every hour (3600s) or on the last epoch
            current_time = time.time()
            elapsed = current_time - last_save_time
            should_save = (elapsed >= 3600) or (epoch == args.epochs - 1)
            
            if should_save:
                last_save_time = current_time
                print(f"Time since last save: {elapsed:.2f}s. Triggering checkpoint save.")

            rnad_bridge.training_worker.perform_offline_training(save_checkpoint=should_save)
            
            # Increment step_count after saving so next save has a unique filename step index
            if should_save:
                rnad_bridge.training_worker.step_count += 1
        
        print("\n--- Offline Training Finished Successfully ---")

    except Exception as e:
        print(f"\n--- Error during offline training ---\n{e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
