import os
import sys
import argparse
import glob
import time
import traceback

# Set environment variable to skip default R-NaD initialization
os.environ["SKIP_RNAD_INIT"] = "1"

# Add the R-NaD directory to sys.path
R_NAD_DIR = os.path.dirname(os.path.abspath(__file__))
if R_NAD_DIR not in sys.path:
    sys.path.insert(0, R_NAD_DIR)

# Standard imports after setting skip flag
import rnad_bridge
from download_human_data import download_human_data

def main():
    # Download latest human play data from Discord before starting
    try:
        download_human_data()
    except Exception as e:
        print(f"Warning: Failed to download human data: {e}")

    parser = argparse.ArgumentParser(description="Offline Training for R-NaD StS2")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint .pkl to resume from")
    parser.add_argument("--epochs", type=int, default=100000, help="Number of passes through all found trajectories")
    parser.add_argument("--save_interval", type=int, default=1, help="Frequency (in epochs) to save checkpoints and log to MLflow")
    args = parser.parse_args()

    print(f"--- Starting Offline Training (Epochs: {args.epochs}) ---")
    
    # Auto-detect latest checkpoint if not specified
    if args.checkpoint is None:
        checkpoint_dir = os.path.join(R_NAD_DIR, "checkpoints")
        if os.path.exists(checkpoint_dir):
            # Recursively search for checkpoint_*.pkl
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "**", "checkpoint_*.pkl"), recursive=True)
            if checkpoints:
                # Find the latest by modification time
                args.checkpoint = max(checkpoints, key=os.path.getmtime)
                print(f"Auto-detected latest checkpoint: {args.checkpoint}")
            else:
                print("No checkpoints found in 'checkpoints/' directory.")
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
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
            
            # Save checkpoint every N epochs or on the last epoch
            should_save = ((epoch + 1) % args.save_interval == 0) or (epoch == args.epochs - 1)
            rnad_bridge.training_worker.perform_offline_training(save_checkpoint=should_save)
        
        print("\n--- Offline Training Finished Successfully ---")

    except Exception as e:
        print(f"\n--- Error during offline training ---\n{e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
