import os
import sys
import ctypes
import json
import random
import threading
import time
import traceback

# Ensure stdout/stderr are unbuffered and also log to a file
import io
LOG_FILE = "/tmp/rnad_bridge.log"
log_handle = open(LOG_FILE, "a", buffering=1) # line buffered

class Logger:
    def __init__(self, original, file_handle):
        self.original = original
        self.file_handle = file_handle
    def write(self, message):
        if self.original:
            self.original.write(message)
        self.file_handle.write(message)
        self.file_handle.flush()
    def flush(self):
        if self.original:
            self.original.flush()
        self.file_handle.flush()

if not hasattr(sys.stdout, "is_rnad_logger"):
    sys.stdout = Logger(sys.stdout, log_handle)
    sys.stderr = Logger(sys.stderr, log_handle)
    # Using a list to avoid attribute error on some builtin streams if they don't allow setting attributes
    # but sys.stdout usually does. Or better, just check type.
    sys.stdout.is_rnad_logger = True

def log(msg):
    # Use direct file write if stdout is weird, but print should be fine now
    print(f"[Python][SM:{id(sys.modules)}][P:{os.getpid()}] {msg}")

log(f"--- Bridge starting at {time.ctime()} ---")
log(f"sys.path: {sys.path[:3]}")
VENV_PATH = "/home/ubuntu/src/R-NaD-StS2/R-NaD/venv/lib/python3.12/site-packages"
if VENV_PATH not in sys.path:
    sys.path.insert(0, VENV_PATH)
    print(f"[Python] Added venv to sys.path: {VENV_PATH}")

# Fix for "undefined symbol: PyObject_SelfIter" in embedded environments
# This forces libpython to load with RTLD_GLOBAL, making its symbols available to C-extensions like NumPy
try:
    # Try common library names
    for libname in ["libpython3.12.so.1.0", "libpython3.12.so.1", "libpython3.12.so"]:
        try:
            ctypes.CDLL(libname, mode=ctypes.RTLD_GLOBAL)
            print(f"[Python] Successfully loaded {libname} with RTLD_GLOBAL")
            break
        except OSError:
            continue
except Exception as e:
    print(f"[Python] Warning: Could not apply RTLD_GLOBAL hack: {e}")
import time
import socket
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
try:
    from PIL import ImageGrab
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Configure JAX to not preallocate all memory and force CPU if needed
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORMS"] = "cpu"

# Placeholders for deferred imports
jax = None
jnp = None
np = None
RNaDLearner = None
RNaDConfig = None

def do_deferred_imports():
    global jax, jnp, np, RNaDLearner, RNaDConfig
    if jax is None:
        import numpy as np_mod
        import jax as jax_mod
        import jax.numpy as jnp_mod
        from src.rnad import RNaDLearner as Learner, RNaDConfig as Config
        
        jax = jax_mod
        jnp = jnp_mod
        np = np_mod
        RNaDLearner = Learner
        RNaDConfig = Config
        print("[Python] Deferred imports completed.")

# Global state
learning_active = False
# Preserve globals across re-imports if already in sys.modules
if 'rnad_bridge' in sys.modules:
    old_mod = sys.modules['rnad_bridge']
    command_queue = getattr(old_mod, 'command_queue', queue.Queue())
    initialized = getattr(old_mod, 'initialized', False)
    learner = getattr(old_mod, 'learner', None)
    config = getattr(old_mod, 'config', None)
    np = getattr(old_mod, 'np', None)
    log(f"Preserved state from existing rnad_bridge module. Queue size: {command_queue.qsize() if hasattr(command_queue, 'qsize') else 'unknown'}")
else:
    log("rnad_bridge module fresh load.")
    command_queue = queue.Queue()
    initialized = False
    learner = None
    config = None
    np = None

current_seed = None

# Config placeholder (will be initialized in load_model)
# config = None # Removed as it's handled by the preservation logic above
# learner = None # Removed as it's handled by the preservation logic above
rng_key = None

# Trajectory and Training Worker
experience_queue = queue.Queue()
current_trajectory = []

class TrainingWorker(threading.Thread):
    def __init__(self, learner, config):
        super().__init__(daemon=True)
        self.learner = learner
        self.config = config
        self.batch_buffer = []
        self.running = True
        self.step_count = 0

    def run(self):
        print("[Python] TrainingWorker started.")
        while self.running:
            try:
                # Wait for a trajectory segment
                trajectory = experience_queue.get(timeout=1.0)
                self.batch_buffer.append(trajectory)
                
                if len(self.batch_buffer) >= self.config.batch_size:
                    self.perform_update()
                    self.batch_buffer = []
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Python] Error in TrainingWorker: {e}")

    def perform_update(self):
        # Convert buffer of trajectories to a single batch
        # buffer shape: (B, T, ...)
        obs = np.array([ [t['obs'] for t in traj] for traj in self.batch_buffer ]) # (B, T, dim)
        act = np.array([ [t['act'] for t in traj] for traj in self.batch_buffer ]) # (B, T)
        rew = np.array([ [t['rew'] for t in traj] for traj in self.batch_buffer ]) # (B, T)
        log_prob = np.array([ [t['log_prob'] for t in traj] for traj in self.batch_buffer ]) # (B, T)

        # Transpose to (T, B, ...)
        batch = {
            'obs': jnp.array(obs.transpose(1, 0, 2)),
            'act': jnp.array(act.transpose(1, 0)),
            'rew': jnp.array(rew.transpose(1, 0)),
            'log_prob': jnp.array(log_prob.transpose(1, 0))
        }

        metrics = self.learner.update(batch, self.step_count)
        self.step_count += 1
        
        print(f"[Python] Training Step {self.step_count}: Loss={metrics['loss']:.4f}, Policy Loss={metrics['policy_loss']:.4f}, Entropy Alpha={metrics['alpha']:.4f}")
        
        if self.step_count % 10 == 0:
            checkpoint_path = f"/home/ubuntu/src/R-NaD-StS2/R-NaD/checkpoints/checkpoint_{self.step_count}.pkl"
            self.learner.save_checkpoint(checkpoint_path, self.step_count)
            print(f"[Python] Saved checkpoint to {checkpoint_path}")

training_worker = None

def load_model(checkpoint_path=None):
    global learner, rng_key, training_worker, config
    do_deferred_imports()
    
    state_dim = 128
    num_actions = 50
    config = RNaDConfig(batch_size=8, unroll_length=32)
    learner = RNaDLearner(state_dim, num_actions, config)
    rng_key = jax.random.PRNGKey(42)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        step = learner.load_checkpoint(checkpoint_path)
        print(f"[Python] Loaded JAX model from {checkpoint_path} at step {step}")
    else:
        learner.init(rng_key)
        print("[Python] Initialized new JAX model")
    
    # Start training worker if learning behavior is expected
    if training_worker is None:
        training_worker = TrainingWorker(learner, config)
        training_worker.start()

def encode_state(state):
    vec = np.zeros(128, dtype=np.float32)
    vec[3] = state.get("floor", 0) / 50.0  # Normalize floor
    if state.get("type") == "combat":
        vec[0] = 1.0
        player = state.get("player", {})
        vec[1] = player.get("hp", 0) / 100.0
        vec[2] = player.get("block", 0) / 100.0
    return vec # Return numpy for easier handling before jnp

def compute_reward(state, next_state_type=None):
    # Intermediate reward: floor progression
    floor = state.get("floor", 0)
    reward = floor * 0.01
    
    # Terminal rewards
    if next_state_type == "game_over":
        victory = state.get("victory", False)
        if victory:
            reward += 1.0
        else:
            reward -= 1.0
            
    return reward

def predict_action(state_json):
    global command_queue, learning_active, current_seed, current_trajectory, learner, rng_key
    
    try:
        state = json.loads(state_json)
        state_type = state.get("type", "unknown")
        log(f"predict_action called. state_type: {state_type}, command_queue size: {command_queue.qsize()}")
        if not command_queue.empty():
            cmd = command_queue.get_nowait()
            res = json.dumps({"action": "command", "command": cmd})
            log(f"Popped command: {cmd}, returning: {res}")
            print(f"[Python-Bridge] Returning command to Rust: {res}", flush=True)
            return res

        if learner is None:
            load_model()

        state = json.loads(state_json)
        state_type = state.get("type", "unknown")
        state_vec = encode_state(state)
        
        # Inference
        logits, value = learner.network.apply(learner.params, rng_key, state_vec[None, :])
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Sample action
        rng_key, subkey = jax.random.split(rng_key)
        action_idx = jax.random.categorical(subkey, logits).item()
        log_prob = jax.nn.log_softmax(logits)[0, action_idx].item()
        
        # Decide action (random for now if learning_active but model is fresh, 
        # or follow model if we want to exploit)
        # For this implementation, we follow the model's recommendation.
        
        # Mapping action_idx to actual game actions (Placeholder Mapping)
        # Action space: 
        # 0-9: Play cards 0-9
        # 10-14: Use potions 0-4
        # 15: End turn
        # 16-25: Select reward index 0-9
        # 26: Proceed
        # ... and so on.
        
        action = {"action": "wait"} # Default
        
        if state_type == "combat":
            hand = state.get("hand", [])
            playable_cards = [c for c in hand if c.get("isPlayable")]
            if action_idx < len(hand) and hand[action_idx].get("isPlayable"):
                action = {"action": "play_card", "card_id": hand[action_idx].get("id")}
            elif action_idx == 15:
                action = {"action": "end_turn"}
            else:
                # Heuristic fallback if model picks invalid action
                if playable_cards:
                    c = random.choice(playable_cards)
                    action = {"action": "play_card", "card_id": c.get("id")}
                else:
                    action = {"action": "end_turn"}
        
        elif state_type == "rewards":
            rewards = state.get("rewards", [])
            if 16 <= action_idx < 16 + len(rewards):
                action = {"action": "select_reward", "index": action_idx - 16}
            elif state.get("can_proceed"):
                action = {"action": "proceed"}
            else:
                action = {"action": "wait"}

        # Trajectory collection
        if learning_active:
            reward = compute_reward(state, state_type)
            current_trajectory.append({
                "obs": state_vec,
                "act": int(action_idx),
                "rew": float(reward),
                "log_prob": float(log_prob)
            })
            
            if len(current_trajectory) >= config.unroll_length:
                experience_queue.put(list(current_trajectory))
                current_trajectory = []

        if action["action"] == "wait" and state_type != "unknown":
             log(f"Falling back to heuristic for state: {state_type}")
             action = get_heuristic_action(state)

        res = json.dumps(action)
        log(f"Returning action: {res}")
        return res
            
    except Exception as e:
        print(f"[Python] CRITICAL ERROR in predict_action: {e}")
        traceback.print_exc()
        return json.dumps({"action": "error", "message": str(e)})

def get_heuristic_action(state):
    state_type = state.get("type", "unknown")
    if state_type == "combat":
        hand = state.get("hand", [])
        playable_cards = [c for c in hand if c.get("isPlayable")]
        if playable_cards:
            chosen_card = random.choice(playable_cards)
            return {"action": "play_card", "card_id": chosen_card.get("id")}
        return {"action": "end_turn"}
    
    elif state_type == "map":
        next_nodes = state.get("next_nodes", [])
        if next_nodes:
            chosen_node = random.choice(next_nodes)
            return {"action": "select_map_node", "row": chosen_node.get("row"), "col": chosen_node.get("col")}
    
    elif state_type == "rewards":
        rewards = state.get("rewards", [])
        if rewards:
            return {"action": "select_reward", "index": random.choice(rewards).get("index")}
        if state.get("can_proceed"):
            return {"action": "proceed"}
            
    elif state_type == "game_over":
        return {"action": "return_to_main_menu"}
        
    return {"action": "wait"}


class CommandHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global learning_active, command_queue
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == "/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "learning_active": learning_active,
                "queue_size": experience_queue.qsize(),
                "step_count": training_worker.step_count if training_worker else 0
            }).encode())
            
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

        elif parsed_path.path == "/command":
            query_components = parse_qs(parsed_path.query)
            cmd = query_components.get("cmd", [""])[0]
            if cmd:
                command_queue.put(cmd)
                log(f"Queued command: {cmd}")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "queued", "command": cmd}).encode())
            
        elif parsed_path.path == "/screenshot":
            command_queue.put("screenshot")
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "queued"}).encode())
            
        elif parsed_path.path == "/new_game":
            query_components = parse_qs(parsed_path.query)
            seed = query_components.get("seed", [None])[0]
            cmd = f"start_game:{seed}" if seed else "start_game"
            command_queue.put(cmd)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "queued", "command": cmd}).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_server():
    server_address = ('127.0.0.1', 8081)
    log(f"Starting server on {server_address}...")
    try:
        httpd = HTTPServer(server_address, CommandHandler)
        log("Command server listening on port 8081...")
        httpd.serve_forever()
    except Exception as e:
        log(f"Server error: {e}")

def init():
    global initialized
    if initialized:
        return
    print("[Python] rnad_bridge initializing...")
    try:
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        load_model()
        initialized = True
        print("[Python] rnad_bridge initialization complete.")
    except Exception as e:
        print(f"[Python] Critical error during initialization: {e}")

init()
