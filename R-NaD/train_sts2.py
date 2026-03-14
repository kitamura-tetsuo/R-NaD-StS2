import argparse
import logging
import os
import time
import subprocess
import requests
import json
import signal

logging.basicConfig(level=logging.INFO)

def cleanup_processes():
    logging.info("Cleaning up existing processes...")
    try:
        subprocess.run(["pkill", "-9", "SlayTheSpire2"], stderr=subprocess.DEVNULL)
        time.sleep(1)
    except Exception:
        pass

def launch_game():
    logging.info("Launching Slay the Spire 2...")
    game_dir = "/home/ubuntu/.steam/steam/steamapps/common/Slay the Spire 2"
    cmd = ["./SlayTheSpire2", "--verbose"]
    env = os.environ.copy()
    keys_to_remove = [k for k in env if k.startswith("PYTHON") or k.startswith("VIRTUAL_ENV") or k.startswith("LD_") or k.startswith("CONDA_")]
    for k in keys_to_remove:
        env.pop(k, None)
    
    # Force a clean system PATH
    env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    process = subprocess.Popen(
        cmd,
        cwd=game_dir,
        env=env,
        stdout=open("/tmp/sts2_train_stdout.log", "w"),
        stderr=open("/tmp/sts2_train_stderr.log", "w")
    )
    return process

def wait_for_server(url, timeout=30):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()
    
    cleanup_processes()
    process = launch_game()

    try:
        # Wait for game to initialize by checking status endpoint
        if not wait_for_server("http://127.0.0.1:8081/status", timeout=60):
            raise RuntimeError("Game server failed to start.")
        
        # Wait for Godot to fully load the scene tree before sending new_game
        logging.info("Waiting 5 seconds for Game Scene Tree to initialize...")
        time.sleep(5)

        # Enable learning mode
        logging.info("Enabling learning mode via bridge server...")
        requests.get("http://127.0.0.1:8081/start")

        # Start first run
        logging.info("Starting new game...")
        requests.get("http://127.0.0.1:8081/new_game")

        last_state_path = "/tmp/rnad_last_state.json"
        
        while True:
            # Check training status
            try:
                resp = requests.get("http://127.0.0.1:8081/status")
                if resp.status_code == 200:
                    status_data = resp.json()
                    step_count = status_data.get("step_count", 0)
                    queue_size = status_data.get("queue_size", 0)
                    if step_count % 10 == 0:
                        logging.info(f"Training Status: Step {step_count}/{args.max_steps}, Queue Size: {queue_size}")
                    
                    if step_count >= args.max_steps:
                        logging.info("Max steps reached. Training complete.")
                        break
            except Exception as e:
                pass # Server might be temporarily unreachable

            # Check if game over to restart
            if os.path.exists(last_state_path):
                try:
                    with open(last_state_path, "r") as f:
                        content = f.read()
                        if content.strip() != "{}":
                            state = json.loads(content)
                            if state.get("type") == "game_over":
                                logging.info("Game over detected. Restarting game run in 5s...")
                                time.sleep(5)
                                requests.get("http://127.0.0.1:8081/new_game")
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
