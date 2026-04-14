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

def launch_game(checkpoint=None, seed=None, no_speedup=False, route=False, headless=False, offline=False, mask_card_skip=False, learning_mode_multiple_move=False):
    logging.info("Launching Slay the Spire 2 (Recording Mode)...")
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
    
    env = os.environ.copy()
    keys_to_remove = [k for k in env if k.startswith("PYTHON") or k.startswith("VIRTUAL_ENV") or k.startswith("LD_") or k.startswith("CONDA_")]
    for k in keys_to_remove:
        env.pop(k, None)
    
    # Force a clean system PATH
    env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    if checkpoint:
        env["RNAD_CHECKPOINT"] = os.path.abspath(checkpoint)
        logging.info(f"Setting RNAD_CHECKPOINT environment variable to: {env['RNAD_CHECKPOINT']}")

    # --- RECORDING SPECIFIC CONFIG ---
    env["RNAD_RECORD_ONLY"] = "true"
    env["JAX_PLATFORMS"] = "cpu"
    logging.info("Setting RNAD_RECORD_ONLY=true and JAX_PLATFORMS=cpu")
    # ---------------------------------

    if env.get("RNAD_RUN_ID"):
        logging.info(f"Setting RNAD_RUN_ID environment variable to: {env['RNAD_RUN_ID']}")

    if seed:
        env["RNAD_SEED"] = seed
        logging.info(f"Setting RNAD_SEED environment variable to: {seed}")

    if route:
        env["RNAD_ROUTE"] = "true"
        logging.info("Setting RNAD_ROUTE environment variable to: true")

    if mask_card_skip:
        env["RNAD_MASK_CARD_SKIP"] = "true"
        logging.info("Setting RNAD_MASK_CARD_SKIP environment variable to: true")

    if learning_mode_multiple_move:
        env["RNAD_MULTIPLE_MOVE"] = "true"
        logging.info("Setting RNAD_MULTIPLE_MOVE environment variable to: true")

    # Disable recorder-mod when launched from recording script
    env["STS2_DISABLE_RECORDER_MOD"] = "true"
    logging.info("Setting STS2_DISABLE_RECORDER_MOD=true")

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    process = subprocess.Popen(
        cmd,
        cwd=game_dir,
        env=env,
        stdout=open(os.path.join(log_dir, "sts2_record_stdout.log"), "w"),
        stderr=open(os.path.join(log_dir, "sts2_record_stderr.log"), "w")
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
    raise BridgeConnectionError(f"Server at {url} did not start in time.")

def cleanup_mlflow_tmp_dirs(threshold_hours=2):
    tmp_base = "/tmp"
    now = time.time()
    threshold = threshold_hours * 3600
    count = 0
    try:
        if not os.path.exists(tmp_base):
            return
        for item in os.listdir(tmp_base):
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
    except Exception as e:
        logging.warning(f"Error during MLflow temp directory cleanup: {e}")

def cleanup_checkpoints(keep_n=5):
    base_dir = os.path.dirname(__file__)
    checkpoint_base = os.path.join(base_dir, "checkpoints")
    if os.path.exists(checkpoint_base):
        try:
            for run_id in os.listdir(checkpoint_base):
                run_path = os.path.join(checkpoint_base, run_id)
                if not os.path.isdir(run_path):
                    continue
                pkl_files = glob.glob(os.path.join(run_path, "checkpoint_*.pkl"))
                if not pkl_files:
                    continue
                def get_step_local(path):
                    match = re.search(r"checkpoint_(\d+)\.pkl", os.path.basename(path))
                    return int(match.group(1)) if match else -1
                pkl_files.sort(key=get_step_local, reverse=True)
                to_delete = pkl_files[keep_n:]
                for f in to_delete:
                    try:
                        os.remove(f)
                    except Exception:
                        pass
        except Exception:
            pass

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
        time.sleep(2)
    return False

def get_step_from_path(path):
    """Robustly extract step number from a path/filename."""
    if not path:
        return -1
    basename = os.path.basename(path)
    # Matches checkpoint_123.pkl, checkpoint_offline_123.pkl, or directory name step_123
    match = re.search(r"(?:checkpoint_|step_)(?:offline_)?(\d+)", basename)
    return int(match.group(1)) if match else -1

def get_latest_local_checkpoint():
    search_dirs = [
        "/home/ubuntu/src/R-NaD-StS2/R-NaD/checkpoints",
        "/home/ubuntu/.local/share/Steam/steamapps/common/Slay the Spire 2/checkpoints",
        "/home/ubuntu/.steam/steam/steamapps/common/Slay the Spire 2/checkpoints"
    ]
    latest_file = None
    latest_mtime = 0
    latest_step = -1
    latest_run_id = None

    # Prioritize the main project checkpoints directory
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        for root, _, files in os.walk(base_dir):
            for file in files:
                # Look for both regular and offline checkpoints
                if file.endswith(".pkl") and ("checkpoint_" in file or "checkpoint_offline_" in file):
                    full_path = os.path.join(root, file)
                    try:
                        step = get_step_from_path(full_path)
                        mtime = os.path.getmtime(full_path)
                        # Primary: highest step, Secondary: newest mtime
                        if step > latest_step or (step == latest_step and mtime > latest_mtime):
                            latest_step = step
                            latest_mtime = mtime
                            latest_file = full_path
                            
                            # Extract run_id if available in the path
                            match = re.search(r"checkpoints/([0-9a-f]{32})/", full_path)
                            if match:
                                latest_run_id = match.group(1)
                            else:
                                # Fallback regex for 32-char hex anywhere
                                match_hex = re.search(r"([0-9a-f]{32})", full_path)
                                if match_hex:
                                    latest_run_id = match_hex.group(1)
                    except OSError:
                        continue
    
    return latest_file, latest_run_id, latest_mtime, latest_step

def get_latest_mlflow_checkpoint(experiment_name="R-NaD-StS2", local_step=-1, local_run_id=None):
    mlflow.set_tracking_uri("sqlite:////home/ubuntu/src/R-NaD-StS2/mlflow.db")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        return None, None, 0, -1
    
    best_candidate = {
        "step": -1,
        "run_id": None,
        "art_path": None
    }
    
    try:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["attributes.start_time DESC"], max_results=5)
        if runs.empty:
            return None, None, 0, -1
            
        client = mlflow.tracking.MlflowClient()
        for _, run in runs.iterrows():
            run_id = run.run_id
            try:
                artifacts = client.list_artifacts(run_id, "checkpoints")
                if not artifacts:
                    continue
                
                for art in artifacts:
                    step = get_step_from_path(art.path)
                    if step > best_candidate["step"]:
                        best_candidate["step"] = step
                        best_candidate["run_id"] = run_id
                        best_candidate["art_path"] = art.path
                
                # If we found checkpoints in the most recent run, we can stop searching older runs
                if best_candidate["run_id"]:
                    break
            except Exception:
                continue
        
        # --- OPTIMIZATION: ONLY DOWNLOAD IF BETTER THAN LOCAL ---
        if best_candidate["run_id"] and best_candidate["art_path"]:
            if best_candidate["step"] <= local_step:
                # Already have this step or better locally
                return None, None, 0, -1

            # Download to a persistent local directory: checkpoints/<run_id>/
            dst_dir = os.path.join(os.path.dirname(__file__), "checkpoints", best_candidate["run_id"])
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
            
            logging.info(f"Downloading newer MLflow checkpoint: Step {best_candidate['step']} (Run: {best_candidate['run_id']})")
            local_path = client.download_artifacts(best_candidate["run_id"], best_candidate["art_path"], dst_path=dst_dir)
            
            # The download might be a directory 'step_N' containing pkl files
            pkl_files = glob.glob(os.path.join(local_path, "*.pkl"))
            if pkl_files:
                # Sort by mtime just in case there are multiple, but usually there's one
                pkl_files.sort(key=os.path.getmtime, reverse=True)
                mtime = os.path.getmtime(pkl_files[0])
                return pkl_files[0], best_candidate["run_id"], mtime, best_candidate["step"]
                
        return None, None, 0, -1
    except Exception as e:
        logging.debug(f"MLflow checkpoint retrieval failed: {e}")
        return None, None, 0, -1


_last_checkpoint_check_time = 0
_cached_best_checkpoint = (None, None)

def select_best_checkpoint(force_check=False):
    global _last_checkpoint_check_time, _cached_best_checkpoint
    now = time.time()
    # Limit checkpoint checks to once every 30 seconds to avoid spamming MLflow/SQLite
    if not force_check and now - _last_checkpoint_check_time < 30:
        return _cached_best_checkpoint

    local_ckpt, local_run_id, local_mtime, local_step = get_latest_local_checkpoint()
    mlflow_ckpt, mlflow_run_id, mlflow_mtime, mlflow_step = get_latest_mlflow_checkpoint(local_step=local_step, local_run_id=local_run_id)
    
    result = (None, None)
    if local_ckpt and mlflow_ckpt:
        # Prefer higher step count. If same, prefer newest mtime.
        if local_step > mlflow_step:
            result = (local_ckpt, local_run_id)
        elif mlflow_step > local_step:
            result = (mlflow_ckpt, mlflow_run_id)
        else:
            if local_mtime >= mlflow_mtime:
                result = (local_ckpt, local_run_id)
            else:
                result = (mlflow_ckpt, mlflow_run_id)
    elif local_ckpt:
        result = (local_ckpt, local_run_id)
    elif mlflow_ckpt:
        result = (mlflow_ckpt, mlflow_run_id)
    
    _last_checkpoint_check_time = now
    _cached_best_checkpoint = result
    return result

def take_screenshot(reason: str):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_reason = re.sub(r"[^\w]+", "_", reason).strip("_")
        tmp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tmp"))
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir, exist_ok=True)
        resp = requests.get("http://127.0.0.1:8081/screenshot", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            original_path = data.get("path", "")
            if original_path and os.path.exists(original_path):
                new_name = f"{timestamp}_record_{safe_reason}_screenshot.png"
                new_path = os.path.join(os.path.dirname(original_path), new_name)
                os.rename(original_path, new_path)
                logging.info(f"Screenshot saved: {new_path}")
    except Exception:
        pass

def perform_restart(process, current_checkpoint, args):
    logging.info("Performing HARD restart (Recording Mode)...")
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    cleanup_processes()
    latest_ckpt, latest_run_id = select_best_checkpoint()
    checkpoint = latest_ckpt if latest_ckpt else current_checkpoint
    if latest_run_id:
        os.environ["RNAD_RUN_ID"] = latest_run_id
    new_process = launch_game(
        checkpoint=checkpoint, 
        seed=args.seed, 
        no_speedup=args.no_speedup, 
        route=args.route, 
        headless=args.headless, 
        offline=False, 
        mask_card_skip=args.mask_card_skip,
        learning_mode_multiple_move=args.learning_mode_multiple_move
    )
    if not wait_for_server("http://127.0.0.1:8081/status", timeout=300):
        return new_process, checkpoint
    wait_for_bridge_initialization(timeout=300)
    time.sleep(5)
    requests.get("http://127.0.0.1:8081/start", timeout=5)
    new_game_url = "http://127.0.0.1:8081/new_game"
    if args.seed:
        new_game_url += f"?seed={args.seed}"
    requests.get(new_game_url, timeout=5)
    time.sleep(10)
    return new_process, checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no-speedup", action="store_true")
    parser.add_argument("--route", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--mask-card-skip", action="store_true")
    parser.add_argument("--learning-mode-multiple-move", action="store_true")
    args = parser.parse_args()
    
    cleanup_processes()
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint, run_id = select_best_checkpoint()
        if run_id:
            os.environ["RNAD_RUN_ID"] = run_id

    process = launch_game(
        checkpoint=checkpoint, 
        seed=args.seed, 
        no_speedup=args.no_speedup, 
        route=args.route, 
        headless=args.headless, 
        offline=False, 
        mask_card_skip=args.mask_card_skip,
        learning_mode_multiple_move=args.learning_mode_multiple_move
    )

    try:
        if not wait_for_server("http://127.0.0.1:8081/status", timeout=300):
            raise RuntimeError("Game server failed to start.")
        time.sleep(5)
        if not wait_for_bridge_initialization(timeout=300):
            raise RuntimeError("R-NaD bridge initialization timed out.")
        requests.get("http://127.0.0.1:8081/start")
        
        logging.info("Waiting for game client to report continuation status...")
        can_continue = None
        status_wait_start = time.time()
        while time.time() - status_wait_start < 60:
            try:
                status_resp = requests.get("http://127.0.0.1:8081/status", timeout=5).json()
                can_continue = status_resp.get("can_continue")
                if can_continue is not None:
                    break
            except Exception:
                pass
            time.sleep(2)
            
        if can_continue:
            requests.get("http://127.0.0.1:8081/continue_game")
        else:
            new_game_url = "http://127.0.0.1:8081/new_game"
            if args.seed:
                new_game_url += f"?seed={args.seed}"
            requests.get(new_game_url)

        time.sleep(60)
        last_activity_time = time.time()
        last_cleanup_time = 0
        last_new_game_time = time.time()
        last_traj_step = -1
        last_step_change_time = time.time()
        
        while True:
            if time.time() - last_cleanup_time > 3600:
                cleanup_mlflow_tmp_dirs()
                cleanup_checkpoints()
                last_cleanup_time = time.time()

            if process.poll() is not None:
                take_screenshot("record_process_exited")
                process, checkpoint = perform_restart(process, checkpoint, args)
                continue

            try:
                resp = requests.get("http://127.0.0.1:8081/status", timeout=5)
                if resp.status_code == 200:
                    status_data = resp.json()
                    last_activity_time = status_data.get("last_activity_time", time.time())
                    
                    is_main_menu = status_data.get("can_continue") is False
                    is_restoring = status_data.get("is_restoring") is True
                    current_traj_step = status_data.get("trajectory_step", 0)
                    current_step_count = status_data.get("step_count", 0)

                    if current_traj_step != last_traj_step:
                        last_traj_step = current_traj_step
                        last_step_change_time = time.time()

                    # Existing 3-minute activity watchdog
                    if time.time() - last_activity_time > 180:
                        take_screenshot("record_stall_detected")
                        process, checkpoint = perform_restart(process, checkpoint, args)
                        last_step_change_time = time.time() # Reset step timer on restart
                        continue
                    
                    # New 1-minute trajectory-step-growth watchdog
                    if time.time() - last_step_change_time > 60:
                        if not is_main_menu and not is_restoring:
                            logging.warning(f"Trajectory step has not increased for 60 seconds (current: {current_traj_step}). Assuming freeze.")
                            take_screenshot("record_step_stall_detected")
                            process, checkpoint = perform_restart(process, checkpoint, args)
                            last_step_change_time = time.time() # Reset step timer on restart
                            continue

                    # New 100-retry-limit watchdog
                    total_retry_count = status_data.get("total_retry_count", 0)
                    if total_retry_count > 100:
                        logging.warning(f"Total retry count ({total_retry_count}) exceeded 100. Starting new game...")
                        take_screenshot("record_retry_limit_exceeded")
                        process, checkpoint = perform_restart(process, checkpoint, args)
                        last_step_change_time = time.time()
                        continue

                    logging.info(f"Recording... Traj: {status_data.get('traj_size', 0)}, Traj Step: {current_traj_step}, Training Step: {current_step_count}")

                    # Continuous recording & Retry handling
                    is_main_menu = status_data.get("can_continue") is False
                    is_restoring = status_data.get("is_restoring") is True

                    if (is_main_menu or is_restoring) and time.time() - last_new_game_time > 60:
                        # Before starting new game or acting on retry, check if checkpoint updated
                        latest_ckpt, latest_run_id = select_best_checkpoint()
                        if latest_ckpt and latest_ckpt != checkpoint:
                            reason = "retry detected" if is_restoring else "run finished"
                            logging.info(f"New checkpoint detected during {reason}: {latest_ckpt}. Restarting game to load latest model...")
                            process, checkpoint = perform_restart(process, checkpoint, args)
                            last_new_game_time = time.time()
                            last_step_change_time = time.time() # Reset step timer
                            continue
                        
                        if is_main_menu:
                            logging.info("Run finished (or at main menu). Starting new game...")
                            new_game_url = f"{BRIDGE_URL}/new_game"
                            if args.seed:
                                new_game_url += f"?seed={args.seed}"
                            try:
                                requests.get(new_game_url, timeout=5)
                                last_new_game_time = time.time()
                            except Exception as e:
                                logging.error(f"Failed to trigger new game: {e}")

            except Exception as e:
                logging.debug(f"Status check failed: {e}")

            time.sleep(5)

    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        try:
            requests.get("http://127.0.0.1:8081/stop")
        except Exception:
            pass
        process.terminate()
        logging.info("=== Recording Session Finished ===")

if __name__ == "__main__":
    main()
