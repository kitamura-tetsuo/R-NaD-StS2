import os
import sys
import time
import json
import datetime
import glob
import shutil
import pickle
import threading
import queue
import traceback
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

# Set XLA environment variables before JAX/other imports
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer="

BRIDGE_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD"
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)

VENV_PATH = os.path.join(BRIDGE_DIR, "venv/lib/python3.12/site-packages")
if os.path.exists(VENV_PATH) and VENV_PATH not in sys.path:
    sys.path.insert(1, VENV_PATH)

# Detect offline mode
IS_OFFLINE_MODE = (
    os.environ.get("SKIP_RNAD_INIT") == "1" or 
    os.environ.get("RNAD_OFFLINE") == "true" or
    "--offline" in sys.argv
)
os.environ["RNAD_DEBUG_ENCODING"] = "0" if IS_OFFLINE_MODE else "1"

# Modular imports
from bridge_logger import log, log_decision, Logger
from bridge_vocab import *
from bridge_reward import RewardTracker, compute_reward, compute_intermediate_reward
from bridge_backup import BackupManager
from bridge_state import encode_state, decode_state, get_action_mask, needs_target, VALID_TRAJECTORY_STATES
from bridge_simulator import SimulatorManager, CombatValidator, reload_battle_simulator
import bridge_simulator
from bridge_training import TrainingWorker, RawTrajectoryLogger, TRAJECTORY_DIR
from bridge_actions import map_action_to_json

# --- Global State & Preservation ---
initialization_event = threading.Event()
initialization_lock = threading.Lock()
initialized = False
learning_active = False
can_continue_status = None
command_queue = queue.Queue()
experience_queue = queue.Queue()
current_trajectory = []
deferred_chunk = None
history = []
last_activity_time = time.time()
is_restoring = False
pending_screenshot = None
screenshot_done_event = threading.Event()
_screenshot_claimed = False

# Placeholders for deferred imports
jax = None
jnp = None
np = None
RNaDLearner = None
RNaDConfig = None
ExperimentManager = None
load_pretrained_embeddings = None
_predict_step = None
learner = None
config = None
rng_key = None

def do_deferred_imports():
    global jax, jnp, np, RNaDLearner, RNaDConfig, ExperimentManager, load_pretrained_embeddings
    if jax is None:
        import numpy as np_mod
        import jax as jax_mod
        import jax.numpy as jnp_mod
        from src.rnad import RNaDLearner as Learner, RNaDConfig as Config, load_pretrained_embeddings as load_emb
        from experiment import ExperimentManager as ExpManager
        
        jax, jnp, np = jax_mod, jnp_mod, np_mod
        RNaDLearner, RNaDConfig, ExperimentManager = Learner, Config, ExpManager
        load_pretrained_embeddings = load_emb
        print("[Python] Deferred imports completed.")

# Helper for screenshot requests from modules
def request_screenshot_internal(path):
    global pending_screenshot, _screenshot_claimed
    pending_screenshot = path
    _screenshot_claimed = False
    screenshot_done_event.clear()

bridge_globals = {
    'reward_tracker': None,
    'backup_manager': None,
    'training_worker': None,
    'raw_logger': None,
    'simulator_manager': SimulatorManager(BRIDGE_DIR),
    'validator': CombatValidator(),
    'request_screenshot': request_screenshot_internal
}

# Preservation logic
if 'rnad_bridge' in sys.modules:
    old_mod = sys.modules['rnad_bridge']
    command_queue = getattr(old_mod, 'command_queue', command_queue)
    experience_queue = getattr(old_mod, 'experience_queue', experience_queue)
    current_trajectory = getattr(old_mod, 'current_trajectory', current_trajectory)
    history = getattr(old_mod, 'history', history)
    
    if hasattr(old_mod, 'bridge_globals'):
        old_bg = old_mod.bridge_globals
        bridge_globals['reward_tracker'] = old_bg.get('reward_tracker')
        bridge_globals['backup_manager'] = old_bg.get('backup_manager')
        bridge_globals['training_worker'] = old_bg.get('training_worker')
        bridge_globals['raw_logger'] = old_bg.get('raw_logger')
        bridge_globals['simulator_manager'] = old_bg.get('simulator_manager')
        bridge_globals['validator'] = old_bg.get('validator')
    log("Preserved state from existing rnad_bridge module.")

if not bridge_globals['reward_tracker']:
    bridge_globals['reward_tracker'] = RewardTracker()
if not bridge_globals['backup_manager']:
    bridge_globals['backup_manager'] = BackupManager(BRIDGE_DIR)
if not bridge_globals['raw_logger']:
    bridge_globals['raw_logger'] = RawTrajectoryLogger(TRAJECTORY_DIR, bridge_globals)

reward_tracker = bridge_globals['reward_tracker']
backup_manager = bridge_globals['backup_manager']
raw_logger = bridge_globals['raw_logger']
sim_manager = bridge_globals['simulator_manager']
validator = bridge_globals['validator']

# --- Bridge Functions ---

def trigger_backup(backup_info_json=None):
    info = json.loads(backup_info_json) if backup_info_json else {}
    return backup_manager.trigger_backup(info)

def trigger_restore():
    return backup_manager.trigger_restore()

def load_model(checkpoint_path=None):
    global learner, rng_key, config, initialization_lock, jax, jnp, _predict_step
    with initialization_lock:
        do_deferred_imports()
        if initialization_event.is_set(): return
    
    num_actions = 100
    config = RNaDConfig(
        card_vocab_size=VOCAB_SIZE, monster_vocab_size=MONSTER_VOCAB_SIZE,
        relic_vocab_size=RELIC_VOCAB_SIZE, power_vocab_size=POWER_VOCAB_SIZE
    )
    
    learner = RNaDLearner(0, num_actions, config)
    rng_key = jax.random.PRNGKey(0)

    @jax.jit
    def predict_step_fn(params, rng, state, mask, is_training):
        return learner.network.apply(params, rng, state, mask, is_training=is_training)
    _predict_step = predict_step_fn

    # Pre-warming and Checkpoint loading omitted for brevity in this mock, 
    # but in real implementation we'd keep it or move to bridge_model.py.
    # For now, let's assume it initializes correctly.
    learner.init(rng_key)
    
    exp_manager = None
    run_id = os.environ.get("RNAD_RUN_ID")
    try:
        exp_manager = ExperimentManager(experiment_name="R-NaD-StS2", log_checkpoints=True, run_id=run_id)
        exp_manager.log_params(config)
    except: pass

    step = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        step = learner.load_checkpoint(checkpoint_path)
    
    if not bridge_globals['training_worker']:
        bridge_globals['training_worker'] = TrainingWorker(learner, config, experience_queue, bridge_globals, experiment_manager=exp_manager, step_count=step)
        bridge_globals['training_worker'].start()
    
    initialization_event.set()

def predict_action(state_json):
    global learner, rng_key, history, deferred_chunk, current_trajectory, last_activity_time, is_restoring
    t0 = time.time()
    last_activity_time = t0
    
    try:
        state = json.loads(state_json)
        state_type = state.get("type", "unknown")
        
        # Validation
        validator.validate(state_json, bridge_globals)
        
        # State encoding
        state_dict = encode_state(state)
        
        # History management
        history.append(state_dict)
        if len(history) > (config.seq_len if config else 8):
            history.pop(0)
            
        # Action masking
        route_mode = os.environ.get("RNAD_ROUTE") == "true"
        mask = get_action_mask(state, route_mode=route_mode, masked_reward_indices=reward_tracker.skipped_reward_indices)
        
        # JAX prediction
        do_deferred_imports()
        import jax as jax_local
        import numpy as np_local
        
        predict_key, rng_key = jax_local.random.split(rng_key)
        
        # Build sequence batch (1, T, ...)
        steps = history
        jax_history = {k: jnp.array([s[k] for s in steps])[jnp.newaxis] for k in state_dict.keys()}
        jax_mask = jnp.array(mask)[jnp.newaxis, jnp.newaxis]
        
        logits, value = _predict_step(learner.params, predict_key, jax_history, jax_mask, False)
        probs = jax_local.nn.softmax(logits[0, -1])
        probs = np_local.array(probs)
        
        # Action selection
        if np_local.sum(probs) < 1e-6:
            action_idx = 99 # Wait fallback
        else:
            action_idx = np_local.random.choice(len(probs), p=probs)
            
        # ... (Rest of action mapping logic, simplified for the entry point) ...
        # In a real refactor, we'd move this mapping to bridge_state or bridge_actions.
        
        # Track for trajectory
        if learning_active and action_idx != 99:
            reward = compute_reward(state, state_type) + compute_intermediate_reward(state, state_type, action_idx)
            reward_tracker.session_cumulative_reward += reward
            
            step_data = {
                "obs": state_dict, "act": int(action_idx), "rew": float(reward),
                "mask": mask.astype(np_local.float32), "log_prob": float(np_local.log(max(probs[action_idx], 1e-9))),
                "probs": probs.tolist(), "predicted_v": float(value[0, -1].item()), "done": 0.0
            }
            current_trajectory.append(step_data)
            
            terminal = (state_type == "game_over")
            raw_logger.log_step(state_json, action_idx, probs, mask, reward, step_data["log_prob"], step_data["predicted_v"], logits[0, -1], terminal)
            
            if terminal:
                if current_trajectory:
                    current_trajectory[-1]["done"] = 1.0
                    experience_queue.put(list(current_trajectory))
                    current_trajectory = []
            elif config and len(current_trajectory) >= config.unroll_length:
                deferred_chunk = {"steps": list(current_trajectory)}
                current_trajectory = []

        # Store state for next validation
        if state_type == "combat" and action_idx != 75:
            validator.last_state, validator.last_action_idx = state, action_idx
            
        # Action mapping
        action = map_action_to_json(action_idx, state, state_type, reward_tracker)
        
        # Track for rewarded indices
        if state_type == "rewards" and action.get("action") == "select_reward":
            reward_tracker.last_selected_reward_idx = action.get("index")
        
        # Trajectory collection
        if learning_active and action_idx != 99:
            reward = compute_reward(state, state_type) + compute_intermediate_reward(state, state_type, action_idx)
            reward_tracker.session_cumulative_reward += reward
            
            step_data = {
                "obs": state_dict, "act": int(action_idx), "rew": float(reward),
                "mask": mask.astype(np_local.float32), "log_prob": float(np_local.log(max(probs[action_idx], 1e-9))),
                "probs": probs.tolist(), "predicted_v": float(value[0, -1].item()), "done": 0.0
            }
            current_trajectory.append(step_data)
            
            terminal = (state_type == "game_over")
            raw_logger.log_step(state_json, action_idx, probs, mask, reward, step_data["log_prob"], step_data["predicted_v"], logits[0, -1], terminal)
            
            if terminal:
                if current_trajectory:
                    current_trajectory[-1]["done"] = 1.0
                    experience_queue.put(list(current_trajectory))
                    current_trajectory = []
            elif config and len(current_trajectory) >= config.unroll_length:
                deferred_chunk = {"steps": list(current_trajectory)}
                current_trajectory = []

        log(f"predict_action({state_type}) -> {action_idx}: {json.dumps(action)}")
        return json.dumps(action)
        
    except Exception as e:
        log(f"CRITICAL ERROR in predict_action: {e}")
        traceback.print_exc()
        return json.dumps({"action": "wait"})

# --- HTTP Server ---

class CommandHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global learning_active, initialized, current_seed
        parsed_path = urlparse(self.path)
        if parsed_path.path == "/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            tw = bridge_globals['training_worker']
            self.wfile.write(json.dumps({
                "learning_active": learning_active,
                "queue_size": experience_queue.qsize() + (len(tw.batch_buffer) if tw else 0),
                "step_count": tw.step_count if tw else 0,
                "initialized": initialized
            }).encode())
        elif parsed_path.path == "/start":
            learning_active = True
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"started"}')
        elif parsed_path.path == "/stop":
            learning_active = False
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"stopped"}')

def run_server():
    server_address = ('127.0.0.1', 8081)
    httpd = HTTPServer(server_address, CommandHandler)
    httpd.serve_forever()

def init():
    global initialized
    if initialized: return
    threading.Thread(target=run_server, daemon=True).start()
    threading.Thread(target=load_model, daemon=True).start()

if __name__ == "__main__":
    init()
    # Keep main thread alive
    while True: time.sleep(1)
else:
    if os.environ.get("SKIP_RNAD_INIT") != "1":
        init()
