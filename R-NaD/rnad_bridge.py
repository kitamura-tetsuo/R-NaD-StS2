import json
import threading
import time
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
try:
    from PIL import ImageGrab
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("[Python] PIL not found or failed to load. PIL-based screenshots will be disabled.")
import os

# Global state
learning_active = False
command_queue = []
current_seed = None

import random

def predict_action(state_json):
    """
    Called from Rust (GDExtension).
    state_json is the JSON serialized game state from C#.
    """
    global learning_active, command_queue
    
    try:
        # Check for pending commands first
        if command_queue:
            cmd = command_queue.pop(0)
            print(f"[Python] Sending command to game: {cmd}")
            return json.dumps({"action": "command", "command": cmd})

        state = json.loads(state_json)
        state_type = state.get("type", "unknown")
        
        print(f"[Python] predict_action called for state type: {state_type}")

        if state_type == "combat":
            hand = state.get("hand", [])
            playable_cards = [c for c in hand if c.get("isPlayable")]
            if playable_cards:
                chosen_card = random.choice(playable_cards)
                action = {
                    "action": "play_card",
                    "card_id": chosen_card.get("id")
                }
            else:
                action = {"action": "end_turn"}
        
        elif state_type == "map":
            next_nodes = state.get("next_nodes", [])
            if next_nodes:
                chosen_node = random.choice(next_nodes)
                action = {
                    "action": "select_map_node",
                    "row": chosen_node.get("row"),
                    "col": chosen_node.get("col")
                }
            else:
                action = {"action": "wait"}

        elif state_type == "event":
            options = state.get("options", [])
            available_options = [o for o in options if not o.get("is_locked")]
            if available_options:
                chosen_option = random.choice(available_options)
                action = {
                    "action": "select_event_option",
                    "index": chosen_option.get("index")
                }
            else:
                action = {"action": "wait"}

        elif state_type == "rewards":
            rewards = state.get("rewards", [])
            if rewards:
                chosen_reward = random.choice(rewards)
                action = {
                    "action": "select_reward",
                    "index": chosen_reward.get("index")
                }
            elif state.get("can_proceed"):
                action = {"action": "proceed"}
            else:
                action = {"action": "wait"}

        elif state_type == "card_reward":
            cards = state.get("cards", [])
            buttons = state.get("buttons", [])
            if cards:
                chosen_card = random.choice(cards)
                action = {
                    "action": "select_reward_card",
                    "index": chosen_card.get("index")
                }
            elif buttons:
                # If no cards left (e.g. they were all selected), click the first button
                # (likely "Skip" or "Done" or "Back")
                action = {
                    "action": "click_reward_button",
                    "index": 0
                }
            else:
                action = {"action": "wait"}

        elif state_type == "grid_selection":
            cards = state.get("cards", [])
            can_skip = state.get("can_skip", False)
            is_confirming = state.get("is_confirming", False)
            
            if is_confirming:
                action = {"action": "confirm_selection"}
            # 80% chance to pick a card, 20% to skip if allowed
            elif cards and (not can_skip or random.random() < 0.8):
                chosen_card = random.choice(cards)
                action = {
                    "action": "select_grid_card",
                    "index": chosen_card.get("index")
                }
            elif can_skip:
                action = {
                    "action": "select_grid_card",
                    "index": -1
                }
            else:
                action = {"action": "wait"}

        elif state_type == "rest_site":
            options = state.get("options", [])
            available_options = [o for o in options if o.get("is_enabled")]
            if available_options:
                chosen_option = random.choice(available_options)
                action = {
                    "action": "select_rest_site_option",
                    "index": chosen_option.get("index")
                }
            elif state.get("can_proceed"):
                action = {"action": "proceed"}
            else:
                action = {"action": "wait"}

        elif state_type == "shop":
            # Always just proceed past the shop
            if state.get("can_proceed"):
                action = {"action": "proceed"}
            else:
                action = {"action": "wait"}

        elif state_type == "treasure":
            if state.get("has_chest"):
                action = {"action": "open_chest"}
            elif state.get("can_proceed"):
                action = {"action": "proceed"}
            else:
                action = {"action": "wait"}

        elif state_type == "treasure_relics":
            relics = state.get("relics", [])
            if relics:
                chosen_relic = random.choice(relics)
                action = {
                    "action": "select_treasure_relic",
                    "index": chosen_relic.get("index")
                }
            else:
                action = {"action": "wait"}

        elif state_type == "game_over":
            action = {"action": "return_to_main_menu"}

        else:
            action = {"action": "wait"}
            
    except Exception as e:
        print(f"[Python] Error in predict_action: {e}")
        action = {"action": "error", "message": str(e)}
        
    # Write last state to a file for E2E testing
    try:
        with open("/tmp/rnad_last_state.json", "w") as f:
            f.write(str(state_json))
    except:
        pass

    return json.dumps(action)


def set_seed(seed):
    global current_seed
    if seed:
        current_seed = str(seed)
        # Convert seed string to integer for random.seed
        # If it's already an integer-like string, use it, otherwise hash it
        try:
            seed_int = int(current_seed)
        except ValueError:
            import hashlib
            seed_int = int(hashlib.md5(current_seed.encode()).hexdigest(), 16) % (2**32)
        
        random.seed(seed_int)
        print(f"[Python] Random seed set to: {current_seed} (int: {seed_int})")
    else:
        current_seed = None
        random.seed(None)
        print("[Python] Random seed reset to None")


class CommandHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global learning_active, command_queue
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == "/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"learning_active": learning_active}).encode())
            
        elif parsed_path.path == "/start":
            learning_active = True
            print("[Python] Learning started!")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started"}).encode())
            
        elif parsed_path.path == "/stop":
            learning_active = False
            print("[Python] Learning stopped!")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "stopped"}).encode())

        elif parsed_path.path == "/new_game":
            query_components = parse_qs(parsed_path.query)
            seed = query_components.get("seed", [None])[0]
            
            if seed:
                set_seed(seed)
                command_queue.append(f"start_game:{seed}")
            else:
                set_seed(None)
                command_queue.append("start_game")
                
            print(f"[Python] New game requested! Seed: {seed}")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "queued", "command": "start_game", "seed": seed}).encode())
            
        elif parsed_path.path == "/screenshot":
            try:
                # Ensure tmp directory exists in project root
                tmp_dir = "/home/ubuntu/src/R-NaD-StS2/tmp"
                os.makedirs(tmp_dir, exist_ok=True)
                
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}.png"
                filepath = os.path.join(tmp_dir, filename)
                
                # Try Godot-side screenshot first (works in headless)
                print(f"[Python] Requesting Godot-side screenshot to: {filepath}")
                command_queue.append(f"screenshot:{filepath}")
                
                # Wait for the file to appear (max 5 seconds)
                success = False
                for _ in range(50):
                    if os.path.exists(filepath):
                        # Wait a tiny bit more for the file to be fully written
                        time.sleep(0.2)
                        success = True
                        break
                    time.sleep(0.1)
                
                if success:
                    print(f"[Python] Godot screenshot saved to: {filepath}")
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "path": filepath, "method": "godot"}).encode())
                    return

                # Fallback to PIL.ImageGrab
                if HAS_PIL:
                    print("[Python] Godot screenshot timed out / failed. Falling back to PIL.ImageGrab.")
                    screenshot = ImageGrab.grab()
                    screenshot.save(filepath)
                    
                    print(f"[Python] PIL screenshot saved to: {filepath}")
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "path": filepath, "method": "pil"}).encode())
                else:
                    print("[Python] Godot screenshot failed and PIL is not available.")
                    self.send_response(500)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "error", "message": "Godot screenshot failed and PIL is not available"}).encode())
            except Exception as e:
                print(f"[Python] Failed to take screenshot: {e}")
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode())
            
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Disable default logging to keep Godot output clean
        pass

def run_server():
    server_address = ('127.0.0.1', 8081)
    httpd = None
    try:
        httpd = HTTPServer(server_address, CommandHandler)
        print("[Python] Command server listening on port 8081...")
        httpd.serve_forever()
    except Exception as e:
        print(f"[Python] Server error: {e}")
    finally:
        if httpd:
            httpd.server_close()

def init():
    print("[Python] rnad_bridge initializing...")
    
    # Start the local command HTTP server in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    print("[Python] rnad_bridge initialization complete.")

# Initialize when loaded
init()

if __name__ == "__main__":
    # Keep the main thread alive if run as a script
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Python] rnad_bridge stopping...")
