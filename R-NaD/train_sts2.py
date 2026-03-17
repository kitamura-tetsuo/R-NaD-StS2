import argparse
import logging
import os
import time
import subprocess
import requests
import json
import signal
import mlflow
import glob
import re

logging.basicConfig(level=logging.INFO)

def cleanup_processes():
    logging.info("Cleaning up existing processes...")
    try:
        subprocess.run(["pkill", "-9", "SlayTheSpire2"], stderr=subprocess.DEVNULL)
        time.sleep(1)
    except Exception:
        pass

def launch_game(checkpoint=None, seed=None, no_speedup=False):
    logging.info("Launching Slay the Spire 2...")
    game_dir = "/home/ubuntu/.steam/steam/steamapps/common/Slay the Spire 2"
    cmd = ["./SlayTheSpire2", "--gym"]
    if seed:
        cmd.extend(["--seed", seed])
    if no_speedup:
        cmd.append("--no-speedup")
    # cmd = ["./SlayTheSpire2", "--verbose", "--gym"]
    env = os.environ.copy()
    keys_to_remove = [k for k in env if k.startswith("PYTHON") or k.startswith("VIRTUAL_ENV") or k.startswith("LD_") or k.startswith("CONDA_")]
    for k in keys_to_remove:
        env.pop(k, None)
    
    # Force a clean system PATH
    env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    if checkpoint:
        env["RNAD_CHECKPOINT"] = os.path.abspath(checkpoint)
        logging.info(f"Setting RNAD_CHECKPOINT environment variable to: {env['RNAD_CHECKPOINT']}")

    if env.get("RNAD_RUN_ID"):
        logging.info(f"Setting RNAD_RUN_ID environment variable to: {env['RNAD_RUN_ID']}")

    if seed:
        env["RNAD_SEED"] = seed
        logging.info(f"Setting RNAD_SEED environment variable to: {seed}")

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    process = subprocess.Popen(
        cmd,
        cwd=game_dir,
        env=env,
        stdout=open(os.path.join(log_dir, "sts2_train_stdout.log"), "w"),
        stderr=open(os.path.join(log_dir, "sts2_train_stderr.log"), "w")
    )
    return process

def wait_for_server(url, timeout=60):
    logging.info(f"Waiting for server at {url} to become available...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            requests.get(url)
            logging.info("Server is up!")
            return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    logging.error("Server did not start in time.")
    return False

def get_latest_mlflow_run_id(experiment_name="R-NaD-StS2"):
    """Finds the latest run ID in the specified MLflow experiment, regardless of checkpoints."""
    try:
        mlflow.set_tracking_uri("file:///home/ubuntu/src/R-NaD-StS2/mlruns")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return None
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        if not runs.empty:
            return runs.iloc[0].run_id
    except Exception as e:
        logging.warning(f"Failed to fetch latest run ID from MLflow: {e}")
    return None

def get_latest_mlflow_checkpoint(experiment_name="R-NaD-StS2"):
    """Finds the latest checkpoint in the specified MLflow experiment."""
    mlflow.set_tracking_uri("file:///home/ubuntu/src/R-NaD-StS2/mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        logging.info(f"Experiment {experiment_name} not found.")
        return None, None

    try:
        # Search for the latest run in this experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=5
        )
        
        if runs.empty:
            logging.info("No runs found in experiment.")
            return None, None

        # Iterate through runs to find one with checkpoint artifacts
        for _, run in runs.iterrows():
            run_id = run.run_id
            client = mlflow.tracking.MlflowClient()
            artifacts = client.list_artifacts(run_id, "checkpoints")
            
            if artifacts:
                # Sort artifacts by step number to find the latest
                # Artifact names are like 'checkpoints/step_10'
                checkpoint_steps = []
                for art in artifacts:
                    match = re.search(r"step_(\d+)", art.path)
                    if match:
                        checkpoint_steps.append((int(match.group(1)), art.path))
                
                if checkpoint_steps:
                    latest_step, latest_art_path = max(checkpoint_steps, key=lambda x: x[0])
                    logging.info(f"Found latest checkpoint at step {latest_step} in run {run_id}")
                    
                    # Download the artifact
                    local_path = client.download_artifacts(run_id, latest_art_path)
                    # The downloaded path will be a directory containing the .pkl file
                    pkl_files = glob.glob(os.path.join(local_path, "*.pkl"))
                    if pkl_files:
                        return pkl_files[0], run_id
        
        logging.info("No checkpoint artifacts found in recent runs.")
        return None, None
    except Exception as e:
        logging.warning(f"Failed to fetch checkpoint from MLflow: {e}")
        return None, None

def take_screenshot(reason: str):
    """Take a screenshot via the bridge API and save it with a descriptive filename.

    The screenshot is saved to ./tmp/ with a name that includes the timestamp
    and the supplied reason string so the cause of each restart is identifiable.
    """
    try:
        resp = requests.get("http://127.0.0.1:8081/screenshot", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            original_path = data.get("path", "")
            if original_path and os.path.exists(original_path):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                # Sanitize reason so it can safely be part of a filename
                safe_reason = re.sub(r"[^\w]+", "_", reason).strip("_")
                new_name = f"screenshot_{timestamp}_{safe_reason}.png"
                new_path = os.path.join(os.path.dirname(original_path), new_name)
                os.rename(original_path, new_path)
                logging.info(f"Screenshot saved: {new_path}")
            else:
                logging.warning(f"Screenshot API returned unexpected path: {original_path}")
        else:
            logging.warning(f"Screenshot API returned status {resp.status_code}")
    except Exception as e:
        logging.warning(f"Failed to take screenshot before restart: {e}")


def perform_restart(process, current_checkpoint, args):
    """Performs a full restart of the game and re-initializes training state."""
    logging.info("Performing HARD restart...")
    
    checkpoint = current_checkpoint
    # Try to find a newer checkpoint if we are in auto-resume mode
    if not args.checkpoint:
        logging.info("Searching for latest checkpoint for restart...")
        new_checkpoint, run_id = get_latest_mlflow_checkpoint()
        if new_checkpoint:
            logging.info(f"Updating checkpoint for restart: {new_checkpoint}")
            checkpoint = new_checkpoint
            if run_id:
                os.environ["RNAD_RUN_ID"] = run_id
                logging.info(f"Continuing MLflow Run ID: {run_id}")
        else:
            # If no checkpoint found, still try to stay on the same run
            run_id = get_latest_mlflow_run_id()
            if run_id:
                os.environ["RNAD_RUN_ID"] = run_id
                logging.info(f"Continuing MLflow Run ID (no new checkpoint): {run_id}")

    # 1. Save the current trajectory buffer to disk before killing the process
    try:
        resp = requests.get("http://127.0.0.1:8081/save_trajectory", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            saved = data.get("saved_steps", 0)
            saved_q = data.get("saved_queues", 0)
            logging.info(f"Saved trajectory of {saved} steps and {saved_q} queued trajectories before restart.")
    except Exception as e:
        logging.warning(f"Failed to save trajectory before restart: {e}")
    
    # 2. Terminate the game process
    if process:
        logging.info("Terminating game process...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logging.info("Process didn't terminate in time, killing it...")
            process.kill()
    
    # Extra cleanup just in case
    cleanup_processes()
    
    # 3. Relaunch the game
    new_process = launch_game(checkpoint=checkpoint, seed=args.seed, no_speedup=args.no_speedup)
    
    # 4. Wait for server and re-initialize
    if not wait_for_server("http://127.0.0.1:8081/status", timeout=120):
        # We don't want to break here as the main loop will catch the error and try again
        logging.error("Failed to recover: game server didn't start.")
        return new_process, checkpoint
    
    logging.info("Waiting 60 seconds for Game Scene Tree to initialize...")
    time.sleep(60)
    
    logging.info("Re-enabling learning mode...")
    try:
        requests.get("http://127.0.0.1:8081/start", timeout=5)
    except Exception as e:
        logging.error(f"Failed to enable learning mode: {e}")

    # 5. Restore the saved trajectory from before the restart
    try:
        resp = requests.get("http://127.0.0.1:8081/load_trajectory", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            loaded = data.get("loaded_steps", 0)
            loaded_q = data.get("loaded_queues", 0)
            logging.info(f"Loaded trajectory of {loaded} steps and {loaded_q} queued trajectories after restart.")
    except Exception as e:
        logging.warning(f"Failed to load trajectory after restart: {e}")

    logging.info(f"Starting new game{' with seed ' + args.seed if args.seed else ''}...")
    new_game_url = "http://127.0.0.1:8081/new_game"
    if args.seed:
        new_game_url += f"?seed={args.seed}"
    try:
        requests.get(new_game_url, timeout=5)
    except Exception as e:
        logging.error(f"Failed to start new game: {e}")
    
    time.sleep(10)

    return new_process, checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--seed", type=str, default=None, help="Fixed seed for new games")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file (.pkl) to resume from")
    parser.add_argument("--no-speedup", action="store_true", help="Disable game acceleration (Instant mode, preload disabling)")
    args = parser.parse_args()
    
    
    cleanup_processes()
    
    checkpoint = args.checkpoint
    run_id = os.environ.get("RNAD_RUN_ID")
    if not checkpoint:
        logging.info("Searching for latest checkpoint in MLflow...")
        checkpoint, run_id = get_latest_mlflow_checkpoint()
        if checkpoint:
            logging.info(f"Resuming from MLflow checkpoint: {checkpoint} (Run ID: {run_id})")
            os.environ["RNAD_RUN_ID"] = run_id
        else:
            logging.info("No MLflow checkpoint found, searching for latest run ID...")
            run_id = get_latest_mlflow_run_id()
            if run_id:
                logging.info(f"Resuming MLflow Run ID (no checkpoint): {run_id}")
                os.environ["RNAD_RUN_ID"] = run_id
            else:
                logging.info("No MLflow run found, starting fresh.")

    process = launch_game(checkpoint=checkpoint, seed=args.seed, no_speedup=args.no_speedup)

    try:
        # Wait for game to initialize by checking status endpoint
        if not wait_for_server("http://127.0.0.1:8081/status", timeout=120):
            raise RuntimeError("Game server failed to start.")
        
        # Wait for Godot to fully load the scene tree before sending new_game
        logging.info("Waiting 5 seconds for Game Scene Tree to initialize...")
        time.sleep(5)

        # Enable learning mode
        logging.info("Enabling learning mode via bridge server...")
        requests.get("http://127.0.0.1:8081/start")

        # Start first run
        logging.info(f"Starting new game{' with seed ' + args.seed if args.seed else ''}...")
        new_game_url = "http://127.0.0.1:8081/new_game"
        if args.seed:
            new_game_url += f"?seed={args.seed}"
        requests.get(new_game_url)

        # Wait for the first game to actually start and generate a seed
        logging.info("Waiting 20 seconds for first game to initialize...")
        time.sleep(20)

        last_state_path = os.path.join(os.path.dirname(__file__), "logs/rnad_last_state.json")
        
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 30 # 30 * 2s = 60s
        
        last_traj_size = -1
        last_traj_progress_time = time.time()
        
        while True:
            # Check if process is still running
            if process.poll() is not None:
                exit_code = process.poll()
                logging.warning(f"Game process exited with code {exit_code}. Restarting...")
                take_screenshot(f"process_exited_code_{exit_code}")
                process, checkpoint = perform_restart(process, checkpoint, args)
                consecutive_failures = 0
                continue

            # Check training status
            try:
                resp = requests.get("http://127.0.0.1:8081/status", timeout=5)
                if resp.status_code == 200:
                    status_data = resp.json()
                    step_count = status_data.get("step_count", 0)
                    queue_size = status_data.get("queue_size", 0)
                    last_activity_time = status_data.get("last_activity_time", time.time())
                    
                    consecutive_failures = 0

                    if time.time() - last_activity_time > 120:
                        logging.warning(f"Stall detected! No progress for {time.time() - last_activity_time:.1f}s.")
                        take_screenshot("stall_no_activity")
                        process, checkpoint = perform_restart(process, checkpoint, args)
                        last_traj_progress_time = time.time()
                        last_traj_size = -1
                        continue

                    queue_size = status_data.get("queue_size", 0)
                    traj_size = status_data.get("traj_size", 0)
                    unroll_length = status_data.get("unroll_length", 0)
                    batch_size = status_data.get("batch_size", 0)
                    
                    # 20s trajectory progress stall detection
                    if traj_size != last_traj_size:
                        last_traj_size = traj_size
                        last_traj_progress_time = time.time()
                        
                    if time.time() - last_traj_progress_time > 20:
                        logging.warning(f"Stall detected! Trajectory size ({traj_size}) hasn't changed for {time.time() - last_traj_progress_time:.1f}s.")
                        take_screenshot(f"stall_traj_size_{traj_size}")
                        process, checkpoint = perform_restart(process, checkpoint, args)
                        last_traj_progress_time = time.time()
                        last_traj_size = -1
                        continue

                    logging.info(f"Training Status: Step {step_count}/{args.max_steps}, Queue: {queue_size}/{batch_size}, Traj: {traj_size}/{unroll_length}")
                    
                    if step_count >= args.max_steps:
                        logging.info("Max steps reached. Training complete.")
                        break
                else:
                    logging.warning(f"Status endpoint returned {resp.status_code}")
                    consecutive_failures += 1
            except (requests.exceptions.RequestException, Exception) as e:
                consecutive_failures += 1
                if consecutive_failures % 5 == 0:
                    logging.warning(f"Communication error ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logging.warning(f"Server unreachable for {consecutive_failures * 2}s. Triggering restart...")
                take_screenshot("server_unreachable")
                process, checkpoint = perform_restart(process, checkpoint, args)
                consecutive_failures = 0
                continue

            # Check if game over to restart
            if os.path.exists(last_state_path):
                try:
                    with open(last_state_path, "r") as f:
                        content = f.read()
                        if content.strip() != "{}":
                            state = json.loads(content)
                            if state.get("type") == "game_over":
                                logging.info(f"Game over detected. Restarting game run{' with seed ' + args.seed if args.seed else ''} in 5s...")
                                # time.sleep(5)
                                new_game_url = "http://127.0.0.1:8081/new_game"
                                if args.seed:
                                    new_game_url += f"?seed={args.seed}"
                                requests.get(new_game_url)
                                # Remove the game_over state so we don't spam restarts
                                os.remove(last_state_path)
                except Exception:
                    pass

            time.sleep(2)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        logging.info("Stopping learning mode and terminating game...")
        try:
            requests.get("http://127.0.0.1:8081/stop")
        except Exception:
            pass
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        logging.info("=== Training Session Finished ===")

if __name__ == "__main__":
    main()
