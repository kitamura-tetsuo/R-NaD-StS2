import requests
import json
import time
import subprocess
import os
import signal
import argparse

def cleanup_processes():
    print("[Test] Cleaning up existing processes...")
    try:
        # Kill the game
        subprocess.run(["pkill", "-9", "SlayTheSpire2"], stderr=subprocess.DEVNULL)
        # Kill other python test scripts, but not ourselves
        current_pid = os.getpid()
        result = subprocess.run(["pgrep", "-f", "e2e_"], stdout=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            pid = int(line.strip())
            if pid != current_pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        time.sleep(1)
    except Exception:
        pass

def test_random_play(max_turns=50):
    print("=== Starting E2E Random Play Test (Headed) ===")
    cleanup_processes()
    
    last_state_path = "/tmp/rnad_last_state.json"
    if os.path.exists(last_state_path):
        os.remove(last_state_path)

    # 1. Launch Slay the Spire 2
    print("[Test] Launching Slay the Spire 2...")
    game_dir = "/home/ubuntu/.steam/steam/steamapps/common/Slay the Spire 2"
    cmd = ["./SlayTheSpire2", "--verbose", "--fastmp"]
    
    # Run in headed mode as requested
    process = subprocess.Popen(
        cmd,
        cwd=game_dir,
        stdout=open("/tmp/sts2_stdout.log", "w"),
        stderr=open("/tmp/sts2_stderr.log", "w")
    )
    
    try:
        # Wait for game to initialize
        print("[Test] Waiting for game to initialize (5s)...")
        time.sleep(5)
        
        # 2. Trigger New Game
        print("[Test] Requesting new game...")
        try:
            response = requests.get("http://127.0.0.1:8081/new_game")
            print(f"[Test] New game response: {response.json()}")
        except Exception as e:
            print(f"[Test] Failed to request new game: {e}")
            return

        # 3. Monitor random play
        print(f"[Test] Monitoring random play for {max_turns} turns...")
        
        turn_count = 0
        last_mtime = 0
        
        while turn_count < max_turns:
            if os.path.exists(last_state_path):
                mtime = os.path.getmtime(last_state_path)
                if mtime > last_mtime:
                    last_mtime = mtime
                    with open(last_state_path, "r") as f:
                        content = f.read()
                        if content == "{}":
                            continue
                        
                        state = json.loads(content)
                        state_type = state.get("type", "unknown")
                        
                        print(f"[Test] Turn {turn_count}: State detected -> {state_type}")
                        if state_type == "combat":
                            hp = state.get("player", {}).get("hp")
                            energy = state.get("player", {}).get("energy")
                            hand_size = len(state.get("hand", []))
                            print(f"       Combat: HP={hp}, Energy={energy}, Hand={hand_size}")
                        elif state_type == "map":
                            current = state.get("current_pos")
                            next_nodes = len(state.get("next_nodes", []))
                            print(f"       Map: Current={current}, Next options={next_nodes}")
                        elif state_type == "event":
                            title = state.get("title")
                            print(f"       Event: {title}")
                        elif state_type == "rewards":
                            rewards = len(state.get("rewards", []))
                            print(f"       Rewards: {rewards} available")
                        elif state_type == "game_over":
                            victory = state.get("victory", False)
                            print(f"       GAME OVER: Victory={victory}")
                            # Give it a moment and then wrap up or reset
                            time.sleep(5)
                            print("[Test] SUCCESS: Run completed (Victory or Game Over).")
                            return
                        
                        turn_count += 1
            
            time.sleep(5)
            if turn_count % 5 == 0 and turn_count > 0:
                print(f"[Test] Progress: {turn_count} events recorded...")

        print("[Test] SUCCESS: Random play goal reached.")

    except KeyboardInterrupt:
        print("[Test] Interrupted by user.")
    finally:
        # 4. Terminate the game
        print("[Test] Terminating Slay the Spire 2...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("=== Test Finished ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=50, help="Maximum number of turns/events to record")
    args = parser.parse_args()
    
    test_random_play(max_turns=args.turns)
