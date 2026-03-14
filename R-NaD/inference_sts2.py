import requests
import json
import time
import subprocess
import os
import signal
import argparse
import random
import string

def cleanup_processes():
    print("[Inference] Cleaning up existing processes...")
    try:
        subprocess.run(["pkill", "-9", "SlayTheSpire2"], stderr=subprocess.DEVNULL)
    except Exception:
        pass

def play_game(seed=None):
    print(f"=== Starting R-NaD Inference (Headed) Seed={seed} ===")
    cleanup_processes()
    
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    last_state_path = os.path.join(log_dir, "rnad_last_state.json")
    if os.path.exists(last_state_path):
        os.remove(last_state_path)

    # 1. Launch Slay the Spire 2
    # The launch.sh script already handles libpython LD_PRELOAD
    print("[Inference] Launching Slay the Spire 2...")
    root_dir = "/home/ubuntu/src/R-NaD-StS2"
    
    # We use launch.sh to ensure environment is correct
    process = subprocess.Popen(
        ["/bin/bash", "./launch.sh"],
        cwd=root_dir,
        stdout=open(os.path.join(log_dir, "sts2_inference_stdout.log"), "w"),
        stderr=open(os.path.join(log_dir, "sts2_inference_stderr.log"), "w")
    )
    
    try:
        # 2. Wait for Bridge Server
        print("[Inference] Waiting for bridge server to start (8081)...")
        server_ready = False
        for _ in range(30): # 30 seconds timeout
            try:
                response = requests.get("http://127.0.0.1:8081/status", timeout=1)
                if response.status_code == 200:
                    server_ready = True
                    print("[Inference] Bridge server is READY.")
                    break
            except requests.exceptions.RequestException:
                time.sleep(1)
        
        if not server_ready:
            print("[Inference] ERROR: Bridge server timed out.")
            return

        # 2.5 Wait for Main Menu
        print("[Inference] Waiting for game to reach Main Menu (type: none)...")
        wait_start = time.time()
        at_menu = False
        while time.time() - wait_start < 30:
            try:
                response = requests.get("http://127.0.0.1:8081/status", timeout=1)
                # The bridge returns 'none' when no run is active (main menu)
                # Note: /status returns training status, we need state from /predict_action or similar
                # Actually, let's just wait a bit more and check if we can get a state.
                # But wait, we don't have a direct /state API, the mod calls US.
                # However, the bridge log shows it's already polling.
                
                # Let's just wait an extra 10 seconds for the game to settle.
                time.sleep(10)
                at_menu = True
                break
            except:
                time.sleep(1)
        
        if not at_menu:
            print("[Inference] WARNING: Could not confirm Main Menu state, proceeding anyway...")

        # 3. Trigger New Game
        print("[Inference] Requesting new game...")
        try:
            # Add a bit of extra cushion
            time.sleep(5)
            url = "http://127.0.0.1:8081/new_game"
            if seed:
                url += f"?seed={seed}"
            response = requests.get(url)
            print(f"[Inference] New game response: {response.json()}")
        except Exception as e:
            print(f"[Inference] Failed to request new game: {e}")
            return

        # 4. Monitor state
        print("[Inference] Monitoring game state. Game window should be visible.")
        
        last_mtime = 0
        while True:
            if os.path.exists(last_state_path):
                mtime = os.path.getmtime(last_state_path)
                if mtime > last_mtime:
                    last_mtime = mtime
                    with open(last_state_path, "r") as f:
                        try:
                            state = json.loads(f.read())
                            state_type = state.get("type", "unknown")
                            if state_type == "combat":
                                hp = state.get("player", {}).get("hp")
                                floor = state.get("floor", 0)
                                print(f"[Inference] Floor {floor} | Combat: HP={hp}")
                            elif state_type == "game_over":
                                victory = state.get("victory", False)
                                print(f"[Inference] GAME OVER: Victory={victory}")
                                time.sleep(10)
                                break
                        except:
                            continue
            
            if process.poll() is not None:
                print("[Inference] Game process exited.")
                break
                
            time.sleep(2)

    except KeyboardInterrupt:
        print("[Inference] Interrupted by user.")
    finally:
        print("[Inference] Terminating game...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("=== Inference Finished ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=str, default=None, help="Fixed seed for reproducibility")
    args = parser.parse_args()
    
    if args.seed is None:
        args.seed = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    
    play_game(seed=args.seed)
