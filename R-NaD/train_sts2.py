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
import shutil

logging.basicConfig(level=logging.INFO)

BRIDGE_URL = "http://127.0.0.1:8081"

class BridgeConnectionError(Exception):
    """Exception raised when the bridge server is unreachable."""
    pass

def cleanup_processes():
    logging.info("Cleaning up existing processes...")
    try:
        subprocess.run(["pkill", "-9", "SlayTheSpire2"], stderr=subprocess.DEVNULL)
        time.sleep(1)
    except Exception:
        pass

def launch_game(checkpoint=None, seed=None, no_speedup=False, route=False, headless=False, offline=False, mask_card_skip=False):
    logging.info("Launching Slay the Spire 2...")
    game_dir = "/home/ubuntu/.steam/steam/steamapps/common/Slay the Spire 2"
    cmd = ["./SlayTheSpire2", "--gym", "--train"]
    if seed:
        cmd.extend(["--seed", seed])
    if no_speedup:
        cmd.append("--no-speedup")
    if headless:
        cmd.append("--headless")
    if offline:
        cmd.append("--offline")
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

    if route:
        env["RNAD_ROUTE"] = "true"
        logging.info("Setting RNAD_ROUTE environment variable to: true")

    if offline:
        env["RNAD_OFFLINE"] = "true"
        logging.info("Setting RNAD_OFFLINE environment variable to: true")

    if mask_card_skip:
        env["RNAD_MASK_CARD_SKIP"] = "true"
        logging.info("Setting RNAD_MASK_CARD_SKIP environment variable to: true")

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

def launch_ui():
    logging.info("Launching Real-time Inference UI (FastAPI + React)...")
    base_dir = os.path.dirname(__file__)
    server_script = os.path.join(base_dir, "live_ui_server.py")
    web_dir = os.path.join(base_dir, "live_ui_web")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Launch FastAPI Backend
    python_path = os.path.join(base_dir, "venv/bin/python")
    backend_process = subprocess.Popen(
        [python_path, server_script],
        stdout=open(os.path.join(log_dir, "ui_backend_stdout.log"), "w"),
        stderr=open(os.path.join(log_dir, "ui_backend_stderr.log"), "w")
    )
    
    # 2. Launch Vite Frontend
    # Use npm run dev and specify the port to be sure
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", "5173", "--host"],
        cwd=web_dir,
        stdout=open(os.path.join(log_dir, "ui_frontend_stdout.log"), "w"),
        stderr=open(os.path.join(log_dir, "ui_frontend_stderr.log"), "w")
    )
    
    return [backend_process, frontend_process]

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
    raise BridgeConnectionError(f"Server at {url} did not start in time.")

def check_disk_space(path, threshold=0.1):
    """Returns True if the disk space on the given path is below the threshold."""
    try:
        usage = shutil.disk_usage(path)
        free_ratio = usage.free / usage.total
        if free_ratio <= threshold:
            logging.error(f"Low disk space: {usage.free / (1024**3):.2f}GB free ({free_ratio*100:.1f}%) on {path}")
            return True
        return False
    except Exception as e:
        logging.warning(f"Failed to check disk space on {path}: {e}")
        return False

def cleanup_mlflow_tmp_dirs(threshold_hours=2):
    """
    Scan /tmp for orphaned MLflow temporary directories (tmpXXXXXX)
    and remove those older than threshold_hours.
    """
    tmp_base = "/tmp"
    now = time.time()
    threshold = threshold_hours * 3600
    count = 0
    try:
        if not os.path.exists(tmp_base):
            return
        for item in os.listdir(tmp_base):
            # MLflow uses tempfile.mkdtemp which typically creates 'tmpXXXXXXXX'
            if item.startswith("tmp") and len(item) >= 8:
                item_path = os.path.join(tmp_base, item)
                try:
                    if os.path.isdir(item_path):
                        mtime = os.path.getmtime(item_path)
                        if now - mtime > threshold:
                            logging.info(f"Cleaning up orphaned MLflow directory: {item_path} (age: {(now-mtime)/3600:.1f}h)")
                            shutil.rmtree(item_path, ignore_errors=True)
                            count += 1
                except (OSError, PermissionError):
                    continue
        if count > 0:
            logging.info(f"Cleanup finished. Removed {count} orphaned directories.")
    except Exception as e:
        logging.warning(f"Error during MLflow temp directory cleanup: {e}")

def wait_for_bridge_initialization(timeout=300):
    logging.info("Waiting for R-NaD bridge to initialize (pre-warm JAX)...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"{BRIDGE_URL}/status", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("initialized", False):
                    logging.info("R-NaD bridge fully initialized.")
                    return True
        except requests.exceptions.RequestException:
            pass
        except Exception as e:
            logging.error(f"Unexpected error in wait_for_bridge_initialization: {e}")
            # Don't loop forever on non-request errors (like NameError)
            return False
        time.sleep(2)
    return False

def get_latest_local_checkpoint():
    """Finds the most recently modified checkpoint file in common local directories."""
    search_dirs = [
        "/home/ubuntu/src/R-NaD-StS2/R-NaD/checkpoints",
        "/home/ubuntu/.local/share/Steam/steamapps/common/Slay the Spire 2/checkpoints",
        "/home/ubuntu/.steam/steam/steamapps/common/Slay the Spire 2/checkpoints"
    ]
    
    latest_file = None
    latest_mtime = 0
    
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
            
        # Search recursively for .pkl files
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".pkl") and "checkpoint_" in file:
                    full_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(full_path)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_file = full_path
                    except OSError:
                        continue
    
    if latest_file:
        # Extract run_id from path if it follows checkpoints/<run_id>/ pattern
        run_id = None
        match = re.search(r"checkpoints/([0-9a-f]{32})/", latest_file)
        if match:
            run_id = match.group(1)
        return latest_file, run_id, latest_mtime
        
    return None, None, 0

def get_latest_mlflow_checkpoint(experiment_name="R-NaD-StS2"):
    """Finds the latest checkpoint in the specified MLflow experiment, returning the one with the highest step number among the recent runs."""
    mlflow.set_tracking_uri("file:///home/ubuntu/src/R-NaD-StS2/mlruns")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        logging.info(f"Experiment {experiment_name} not found.")
        return None, None, 0
    
    latest_step_global = -1
    latest_pkl_path = None
    latest_run_id = None
    latest_mtime = 0

    try:
        # Search for the latest runs in this experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=5
        )
        
        if runs.empty:
            logging.info("No runs found in experiment.")
            return None, None, 0

        # Iterate through runs to find the latest checkpoint
        for _, run in runs.iterrows():
            run_id = run.run_id
            client = mlflow.tracking.MlflowClient()
            artifacts = client.list_artifacts(run_id, "checkpoints")
            
            if artifacts:
                for art in artifacts:
                    match = re.search(r"step_(\d+)", art.path)
                    if match:
                        step = int(match.group(1))
                        # For MLflow, we still use the highest step number within a run to find its 'latest'
                        # but we will compare the local vs MLflow later using mtime.
                        
                        # Download if it's potentially the newest
                        local_path = client.download_artifacts(run_id, art.path)
                        pkl_files = glob.glob(os.path.join(local_path, "*.pkl"))
                        if pkl_files:
                            mtime = os.path.getmtime(pkl_files[0])
                            # In MLflow we still prefer higher step count if multiple checkpoints exist in the SAME run
                            # because mtime of downloaded artifacts might be reset to 'now'.
                            # However, across runs we will rely on the start_time order.
                            if step > latest_step_global:
                                latest_step_global = step
                                latest_pkl_path = pkl_files[0]
                                latest_run_id = run_id
                                latest_mtime = mtime
                
                # If we found at least one checkpoint in the latest run that has artifacts, we stop here
                # to avoid picking a higher step count from an older run.
                if latest_pkl_path:
                    break
        
        return latest_pkl_path, latest_run_id, latest_mtime
    except Exception as e:
        logging.warning(f"Failed to fetch checkpoint from MLflow: {e}")
        return None, None, 0

def select_best_checkpoint():
    """Selects the best checkpoint by comparing local files and MLflow artifacts using mtime."""
    local_ckpt, local_run_id, local_mtime = get_latest_local_checkpoint()
    mlflow_ckpt, mlflow_run_id, mlflow_mtime = get_latest_mlflow_checkpoint()
    
    if local_ckpt and mlflow_ckpt:
        if local_mtime >= mlflow_mtime:
            logging.info(f"Selecting local checkpoint (mtime: {time.ctime(local_mtime)}): {local_ckpt}")
            return local_ckpt, local_run_id
        else:
            logging.info(f"Selecting MLflow checkpoint (mtime: {time.ctime(mlflow_mtime)}): {mlflow_ckpt}")
            return mlflow_ckpt, mlflow_run_id
    elif local_ckpt:
        logging.info(f"Selecting local checkpoint: {local_ckpt}")
        return local_ckpt, local_run_id
    elif mlflow_ckpt:
        logging.info(f"Selecting MLflow checkpoint: {mlflow_ckpt}")
        return mlflow_ckpt, mlflow_run_id
    
    return None, None

def save_last_n_lines(src_path, dst_path, n=100):
    """Save the last n lines of a file to a new destination."""
    try:
        if not os.path.exists(src_path):
            return False
        with open(src_path, "r", errors="replace") as f:
            lines = f.readlines()
            last_lines = lines[-n:] if len(lines) > n else lines
        with open(dst_path, "w") as f:
            f.writelines(last_lines)
        return True
    except Exception as e:
        logging.warning(f"Failed to save last {n} lines of {src_path}: {e}")
        return False

def take_screenshot(reason: str):
    """Take a screenshot via the bridge API and save it with a descriptive filename.
    Also saves the last 100 lines of relevant logs.

    The screenshot is saved to ./tmp/ with a name that includes the timestamp
    and the supplied reason string so the cause of each restart is identifiable.
    """
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_reason = re.sub(r"[^\w]+", "_", reason).strip("_")
        tmp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp"))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=True)
            
        # Take screenshot
        try:
            resp = requests.get("http://127.0.0.1:8081/screenshot", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                original_path = data.get("path", "")
                if original_path and os.path.exists(original_path):
                    tmp_dir = os.path.dirname(original_path)
                    new_name = f"{timestamp}_{safe_reason}_screenshot.png"
                    new_path = os.path.join(tmp_dir, new_name)
                    os.rename(original_path, new_path)
                    logging.info(f"Screenshot saved: {new_path}")
                else:
                    logging.warning(f"Screenshot API returned unexpected path: {original_path}")
            else:
                logging.warning(f"Screenshot API returned status {resp.status_code}")
        except Exception as e:
            logging.warning(f"Failed to take screenshot: {e}")

        # Save logs
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        logs_to_save = [
            "rnad_bridge.log",
            "rnad_decisions.log",
            "rnad_last_state.json",
            "sts2_train_stdout.log",
            "sts2_train_stderr.log"
        ]
        
        for log_name in logs_to_save:
            src = os.path.join(log_dir, log_name)
            if not os.path.exists(src):
                continue
                
            dst = os.path.join(tmp_dir, f"{timestamp}_{safe_reason}_{log_name}")
            
            if log_name.endswith(".json"):
                # For JSON, just save the whole file as it's typically small
                shutil.copy(src, dst)
            else:
                save_last_n_lines(src, dst, n=100)
                
            if os.path.exists(dst):
                logging.info(f"Log saved: {dst}")

    except Exception as e:
        logging.warning(f"Failed to process debug info before restart: {e}")

def wait_for_update_to_finish(status_url="http://127.0.0.1:8081/status", max_failures=None):
    """Blocks until is_updating is False and queue is not full in the bridge status."""
    consecutive_failures = 0
    while True:
        try:
            resp = requests.get(status_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                is_updating = data.get("is_updating", False)
                queue_size = data.get("queue_size", 0)
                batch_size = data.get("batch_size", 0)
                
                consecutive_failures = 0 # success, reset counter
                if not is_updating:
                    if batch_size == 0 or queue_size < batch_size:
                        # Log one final check before breaking
                        logging.info("Bridge is not updating. Wait complete.")
                        break
                    logging.info(f"Queue is full ({queue_size}/{batch_size}). Waiting for update to start...")
                else:
                    update_progress = data.get("update_progress", 0)
                    update_total = data.get("update_total", 0)
                    logging.info(f"Bridge is performing an update ({update_progress}/{update_total}). Waiting...")
            else:
                logging.warning(f"Bridge status endpoint returned {resp.status_code}. Retrying...")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            consecutive_failures += 1
            logging.warning(f"Failed to check bridge status ({e}). Retrying...")
            if max_failures and consecutive_failures >= max_failures:
                raise BridgeConnectionError(f"Bridge unreachable after {consecutive_failures} attempts.")
        except Exception as e:
            logging.error(f"Unexpected error while checking bridge status: {e}")
        
        time.sleep(2)

def perform_restart(process, current_checkpoint, args):
    """Performs a full restart of the game and re-initializes training state."""
    # Try to wait for update to finish before restart, but don't hang if bridge is dead
    try:
        wait_for_update_to_finish(max_failures=3)
    except BridgeConnectionError:
        logging.warning("Bridge unreachable, skipping pre-restart update wait.")
    
    logging.info("Performing HARD restart...")
    
    checkpoint = current_checkpoint
    # Try to find a newer checkpoint if we are in auto-resume mode
    # Even if args.checkpoint was provided, on restart we usually want to continue from where we left off (the latest save)
    # unless a specific flag was set to keep using the old one.
    logging.info("Searching for latest checkpoint for restart...")
    new_checkpoint, run_id = select_best_checkpoint()
    if new_checkpoint:
        logging.info(f"Updating checkpoint for restart: {new_checkpoint}")
        checkpoint = new_checkpoint
        if run_id:
            os.environ["RNAD_RUN_ID"] = run_id
            logging.info(f"Continuing MLflow Run ID: {run_id}")
    elif not os.environ.get("RNAD_RUN_ID"):
        # If no checkpoint found, still try to stay on the same run if not already set
        # Re-import mllow briefly or just use subprocess if needed, but we can reuse the logic
        import mlflow
        mlflow.set_tracking_uri("file:///home/ubuntu/src/R-NaD-StS2/mlruns")
        experiment = mlflow.get_experiment_by_name("R-NaD-StS2")
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
            if not runs.empty:
                run_id = runs.iloc[0].run_id
                os.environ["RNAD_RUN_ID"] = run_id
                logging.info(f"Continuing MLflow Run ID (no new checkpoint): {run_id}")
    else:
        logging.info("No active MLflow run found during restart.")

    # 1. Save the current trajectory buffer to disk before killing the process
    try:
        resp = requests.get("http://127.0.0.1:8081/save_trajectory", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            saved = data.get("saved_steps", 0)
            saved_q = data.get("saved_queues", 0)
            logging.info(f"Saved trajectory of {saved} steps and {saved_q} queued trajectories before restart.")
            
            # If the flush triggered a potential update, wait for it before killing the process
            try:
                wait_for_update_to_finish(max_failures=5)
            except BridgeConnectionError:
                logging.warning("Bridge went down while waiting for update after save_trajectory.")
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
    new_process = launch_game(checkpoint=checkpoint, seed=args.seed, no_speedup=args.no_speedup, route=args.route, headless=args.headless, offline=False, mask_card_skip=args.mask_card_skip)
    
    # 4. Wait for server and re-initialize
    if not wait_for_server("http://127.0.0.1:8081/status", timeout=300):
        # We don't want to break here as the main loop will catch the error and try again
        logging.error("Failed to recover: game server didn't start.")
        return new_process, checkpoint
    
    logging.info("Waiting 30 seconds for Game Scene Tree to initialize...")
    time.sleep(30)
    
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
    
    time.sleep(30)

    return new_process, checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--seed", type=str, default=None, help="Fixed seed for new games")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file (.pkl) to resume from")
    parser.add_argument("--no-speedup", action="store_true", help="Disable game acceleration (Instant mode, preload disabling)")
    parser.add_argument("--route", action="store_true", help="Always choose the map room with the smallest index")
    parser.add_argument("--headless", action="store_true", help="Run the game in headless mode")
    parser.add_argument("--offline", action="store_true", help="Enable offline training from trajectories/replays on first launch")
    parser.add_argument("--mask-card-skip", action="store_true", help="Mask Skip in post-combat card rewards")
    parser.add_argument("--ui", action="store_true", help="Launch the Live Inference Monitor (Streamlit)")
    args = parser.parse_args()
    
    cleanup_processes()
    
    checkpoint = args.checkpoint
    run_id = os.environ.get("RNAD_RUN_ID")
    
    # If checkpoint is provided, try to extract run_id from its path if it follows standard pattern
    if checkpoint and not run_id:
        # Checkpoints are often in .../checkpoints/<run_id>/checkpoint_X.pkl
        match = re.search(r"checkpoints/([0-9a-f]{32})/", checkpoint)
        if match:
            run_id = match.group(1)
            logging.info(f"Extracted MLflow Run ID from checkpoint path: {run_id}")
            os.environ["RNAD_RUN_ID"] = run_id

    if not checkpoint:
        logging.info("Searching for latest checkpoint...")
        checkpoint, run_id = select_best_checkpoint()
        if checkpoint:
            logging.info(f"Resuming from detected checkpoint: {checkpoint} (Run ID: {run_id})")
            os.environ["RNAD_RUN_ID"] = run_id
        else:
            # Fallback for just run ID
            import mlflow
            mlflow.set_tracking_uri("file:///home/ubuntu/src/R-NaD-StS2/mlruns")
            experiment = mlflow.get_experiment_by_name("R-NaD-StS2")
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
                if not runs.empty:
                    run_id = runs.iloc[0].run_id
                    logging.info(f"Resuming MLflow Run ID (no checkpoint): {run_id}")
                    os.environ["RNAD_RUN_ID"] = run_id
                else:
                    logging.info("No MLflow run found, starting fresh.")
            else:
                logging.info("No MLflow run found, starting fresh.")

    ui_procs = None
    if args.ui:
        ui_procs = launch_ui()
        logging.info("Real-time UI started at http://localhost:5173")

    process = launch_game(checkpoint=checkpoint, seed=args.seed, no_speedup=args.no_speedup, route=args.route, headless=args.headless, offline=args.offline, mask_card_skip=args.mask_card_skip)

    try:
        # Wait for game to initialize by checking status endpoint
        if not wait_for_server("http://127.0.0.1:8081/status", timeout=300):
            raise RuntimeError("Game server failed to start.")
        
        # Wait for Godot to fully load the scene tree before sending new_game
        logging.info("Waiting 5 seconds for Game Scene Tree to initialize...")
        time.sleep(5)

        # Wait for R-NaD bridge to fully initialize (learner and worker ready)
        if not wait_for_bridge_initialization(timeout=300):
            raise RuntimeError("R-NaD bridge initialization timed out.")

        # Enable learning mode
        logging.info("Enabling learning mode via bridge server...")
        resp = requests.get("http://127.0.0.1:8081/start")
        if resp.status_code != 200:
            logging.error(f"Failed to enable learning: {resp.status_code} {resp.text}")

        # Start first run
        try:
            # Check for offline training
            resp = requests.get("http://127.0.0.1:8081/status", timeout=5)
            if resp.status_code == 200:
                status_data = resp.json()
                step_count = status_data.get("step_count", 0)
                
                traj_dir = "/home/ubuntu/src/R-NaD-StS2/R-NaD/trajectories"
                replay_dir = "/mnt/nas/StS2/replay"
                
                has_traj = os.path.exists(traj_dir) and any(f.endswith(".json") for f in os.listdir(traj_dir) if f.startswith("traj_"))
                has_replay = os.path.exists(replay_dir) and any(f.endswith(".jsonl") for f in os.listdir(replay_dir) if f.startswith("human_play_"))
                
                if args.offline and (has_traj or has_replay):
                    logging.info(f"Step {step_count} detected and data found. Triggering offline training...")
                    off_resp = requests.get("http://127.0.0.1:8081/offline_train", timeout=5)
                    if off_resp.status_code == 200:
                        # Wait for offline training to actually start and finish
                        time.sleep(2)
                        wait_for_update_to_finish(max_failures=300) # Long timeout for offline training
                        logging.info("Offline training complete.")
                    else:
                        logging.error(f"Failed to start offline training: {off_resp.status_code} {off_resp.text}")
            
            # Re-check status one last time to ensure no residual updates
            wait_for_update_to_finish(max_failures=10)
            
            logging.info("Offline training phase cleared. Transitioning to online learning.")
                        
        except BridgeConnectionError:
            logging.warning("Bridge failed during initial update wait. Triggering restart...")
            process, checkpoint = perform_restart(process, checkpoint, args)
            # Re-enter the loop after restart
            
        # Wait for game client to report continuation status
        logging.info("Waiting for game client to report continuation status...")
        can_continue = None
        status_wait_start = time.time()
        while time.time() - status_wait_start < 60: # 60 second timeout
            try:
                status_resp = requests.get("http://127.0.0.1:8081/status", timeout=5).json()
                can_continue = status_resp.get("can_continue")
                if can_continue is not None:
                    break
            except Exception as e:
                pass
            time.sleep(2)
            
        if can_continue:
            logging.info("Continue-able run found. Resuming existing run...")
            requests.get("http://127.0.0.1:8081/continue_game")
        else:
            if can_continue is None:
                logging.warning("Timed out waiting for continuation status. Defaulting to new game.")
            logging.info(f"Starting new game{' with seed ' + args.seed if args.seed else ''}...")
            new_game_url = "http://127.0.0.1:8081/new_game"
            if args.seed:
                new_game_url += f"?seed={args.seed}"
            requests.get(new_game_url)

        # Wait for the first game to actually start and generate a seed
        logging.info("Waiting 60 seconds for first game to initialize...")
        time.sleep(60)

        last_state_path = os.path.join(os.path.dirname(__file__), "logs/rnad_last_state.json")
        
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 30 # 30 * 2s = 60s
        
        last_traj_size = -1
        last_traj_progress_time = time.time()
        last_cleanup_time = 0
        
        while True:
            # Periodically cleanup MLflow tmp dirs (every 1 hour)
            if time.time() - last_cleanup_time > 3600:
                cleanup_mlflow_tmp_dirs()
                last_cleanup_time = time.time()

            # Check if process is still running
            if process.poll() is not None:
                exit_code = process.poll()
                logging.warning(f"Game process exited with code {exit_code}. Restarting...")
                take_screenshot(f"process_exited_code_{exit_code}")
                process, checkpoint = perform_restart(process, checkpoint, args)
                consecutive_failures = 0
                last_traj_progress_time = time.time()
                last_traj_size = -1
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
                    
                    # Check disk space (on the mlruns directory)
                    mlruns_dir = os.path.join(os.path.dirname(__file__), "..", "mlruns")
                    if check_disk_space(mlruns_dir, threshold=0.1):
                        logging.error("FATAL: Insufficient disk space for checkpoints (<10% free). Stopping training.")
                        raise RuntimeError("Low disk space on checkpoint storage.")

                    # Stop if worker encountered a training error
                    worker_error = status_data.get("worker_error")
                    if worker_error:
                        logging.error(f"FATAL: TrainingWorker error detected: {worker_error}")
                        logging.error("Stopping train_sts2.py to prevent unmonitored failures.")
                        raise RuntimeError(f"TrainingWorker error: {worker_error}")

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
                    
                    # 60s trajectory progress stall detection
                    if traj_size != last_traj_size:
                        last_traj_size = traj_size
                        last_traj_progress_time = time.time()
                        
                    if time.time() - last_traj_progress_time > 600:
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
                last_traj_progress_time = time.time()
                last_traj_size = -1
                continue


            time.sleep(2)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        logging.info("Stopping learning mode and terminating game...")
        try:
            requests.get("http://127.0.0.1:8081/stop")
        except Exception:
            pass
        if ui_procs:
            for p in ui_procs:
                p.terminate()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        logging.info("=== Training Session Finished ===")

if __name__ == "__main__":
    main()
