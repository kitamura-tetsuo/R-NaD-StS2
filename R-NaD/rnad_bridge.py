import json
import threading
import time
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# Global state
learning_active = False

def predict_action(state_json):
    """
    Called from Rust (GDExtension).
    state_json is the JSON serialized game state from C#.
    """
    global learning_active
    
    # Process the state...
    print(f"[Python] predict_action called with state: {state_json[:100]}...")
    
    # If learning is active, we might do something different
    if learning_active:
        print("[Python] (Learning mode active)")
    
    # For now, just return a dummy action
    action = {
        "action": "play_card",
        "card_id": "strike",
        "learning": learning_active
    }
    return json.dumps(action)


class CommandHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global learning_active
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
