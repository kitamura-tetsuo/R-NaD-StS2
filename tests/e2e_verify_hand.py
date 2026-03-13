import requests
import json
import time
import subprocess
import os
import signal

def test_hand_extraction():
    print("=== Starting E2E Hand Extraction Test ===")
    
    # Path to the last state file
    last_state_path = "/tmp/rnad_last_state.json"
    if os.path.exists(last_state_path):
        os.remove(last_state_path)

    # 1. Launch Slay the Spire 2 in background
    print("[Test] Launching Slay the Spire 2...")
    game_dir = "/home/ubuntu/.steam/steam/steamapps/common/Slay the Spire 2"
    process = subprocess.Popen(
        ["./SlayTheSpire2", "--verbose"],
        cwd=game_dir,
        stdout=open("/tmp/sts2_stdout.log", "w"),
        stderr=open("/tmp/sts2_stderr.log", "w")
    )
    
    try:
        # Wait for game to initialize (approx 30s)
        time.sleep(30)
        
        # 2. Trigger New Game
        print("[Test] Requesting new game...")
        try:
            response = requests.get("http://127.0.0.1:8081/new_game")
            print(f"[Test] Response: {response.json()}")
        except Exception as e:
            print(f"[Test] Failed to request new game: {e}")
            return

        # 3. Monitor for state file
        print("[Test] Waiting for combat state to be reported...")
        timeout = 180  # seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if os.path.exists(last_state_path):
                with open(last_state_path, "r") as f:
                    content = f.read()
                    if content == "{}":
                        os.remove(last_state_path)
                        continue
                    
                    print("[Test] SUCCESS: Real state extraction verified!")
                    print(f"[Test] State Content: {content}")
                    state = json.loads(content)
                    hand = state.get("hand", [])
                    print(f"[Test] Detected {len(hand)} cards in hand.")
                    for card in hand:
                        print(f"  - {card['name']} ({card['id']})")
                break
            
            time.sleep(2)
            if int(time.time() - start_time) % 10 == 0:
                print(f"[Test] Still waiting... ({int(time.time() - start_time)}s)")
        else:
            print("[Test] FAILED: Timeout waiting for state.")

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
    test_hand_extraction()
