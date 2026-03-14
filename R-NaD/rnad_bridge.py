import os
import sys
import time

# Ensure stdout/stderr are unbuffered and also log to a file
import io
BRIDGE_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD"
LOG_DIR = os.path.join(BRIDGE_DIR, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "rnad_bridge.log")
MAX_LOG_SIZE = 10 * 1024 * 1024 # 10MB
BACKUP_COUNT = 3

class Logger:
    def __init__(self, original, filename):
        self.original = original
        self.filename = filename
        self.file_handle = open(filename, "a", buffering=1)
        self.lock = threading.Lock()

    def _rotate_logs(self):
        self.file_handle.close()
        for i in range(BACKUP_COUNT - 1, 0, -1):
            src = f"{self.filename}.{i}"
            dst = f"{self.filename}.{i+1}"
            if os.path.exists(src):
                if os.path.exists(dst):
                    os.remove(dst)
                os.rename(src, dst)
        
        if os.path.exists(self.filename):
            os.rename(self.filename, f"{self.filename}.1")
            
        self.file_handle = open(self.filename, "w", buffering=1)

    def write(self, message):
        if self.original:
            self.original.write(message)
        
        with self.lock:
            try:
                if os.path.exists(self.filename) and os.path.getsize(self.filename) > MAX_LOG_SIZE:
                    self._rotate_logs()
                self.file_handle.write(message)
                self.file_handle.flush()
            except Exception as e:
                # Fallback to original if file write fails, avoid infinite loops
                if self.original:
                    self.original.write(f"\n[Logger Error] {e}\n")

    def flush(self):
        if self.original:
            self.original.flush()
        with self.lock:
            try:
                self.file_handle.flush()
            except:
                pass

if not hasattr(sys.stdout, "is_rnad_logger"):
    # Note: threading is imported later, but we need it for the lock.
    # Actually, the bridge imports it at line 67. We should move the logger setup
    # after the essential imports or ensure threading is available.
    import threading 
    sys.stdout = Logger(sys.stdout, LOG_FILE)
    sys.stderr = Logger(sys.stderr, LOG_FILE)
    sys.stdout.is_rnad_logger = True

def log(msg):
    # Use direct file write if stdout is weird, but print should be fine now
    print(f"[Python][SM:{id(sys.modules)}][P:{os.getpid()}] {msg}")

log(f"--- Bridge starting at {time.ctime() if 'time' in sys.modules else 'unknown'} ---")
log(f"sys.path: {sys.path[:3]}")

# Add bridge directory to sys.path so we can import our fixer
BRIDGE_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD"
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)

VENV_PATH = "/home/ubuntu/src/R-NaD-StS2/R-NaD/venv/lib/python3.12/site-packages"
if VENV_PATH not in sys.path:
    # Append the virtualenv AFTER the local directory so local modules take precedence
    sys.path.append(VENV_PATH)
    print(f"[Python] Added venv to sys.path: {VENV_PATH}")

# Fix for "undefined symbol: PyObject_SelfIter" and similar dynamic loading issues in embedded environments
# We use a custom C-extension `libpython_fixer` because `import ctypes` itself fails when RTLD_GLOBAL is missing.
try:
    import libpython_fixer
    for libname in ["libpython3.12.so.1.0", "libpython3.12.so.1", "libpython3.12.so"]:
        res = libpython_fixer.fix_libpython(libname)
        if res == "success":
            print(f"[Python] Successfully loaded {libname} with RTLD_GLOBAL via libpython_fixer")
            break
        else:
            print(f"[Python] libpython_fixer failed for {libname}: {res}")
except Exception as e:
    print(f"[Python] Warning: Could not apply RTLD_GLOBAL hack via libpython_fixer: {e}")

# Now we can safely import ctypes and other C-extensions
import ctypes
import json
import random
import threading
import time
import traceback
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
ExperimentManager = None

def do_deferred_imports():
    global jax, jnp, np, RNaDLearner, RNaDConfig, ExperimentManager
    if jax is None:
        import numpy as np_mod
        import jax as jax_mod
        import jax.numpy as jnp_mod
        from src.rnad import RNaDLearner as Learner, RNaDConfig as Config
        from experiment import ExperimentManager as ExpManager
        
        jax = jax_mod
        jnp = jnp_mod
        np = np_mod
        RNaDLearner = Learner
        RNaDConfig = Config
        ExperimentManager = ExpManager
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

current_seed = os.environ.get("RNAD_SEED")
if current_seed:
    log(f"Initialized current_seed from environment: {current_seed}")

# Config placeholder (will be initialized in load_model)
# config = None # Removed as it's handled by the preservation logic above
# learner = None # Removed as it's handled by the preservation logic above
rng_key = None

# Trajectory and Training Worker
experience_queue = queue.Queue()
current_trajectory = []

# Synchronous screenshot signaling
pending_screenshot = None
screenshot_done_event = threading.Event()
_screenshot_claimed = False  # True once _Process has picked up the current request

_poll_data = [0]
def check_screenshot_request():
    global pending_screenshot, _screenshot_claimed
    _poll_data[0] += 1
    if pending_screenshot and _poll_data[0] % 10 == 0:
        log(f"check_screenshot_request polled {_poll_data[0]} times. pending={pending_screenshot}")
    # Only return the path if not already claimed by another caller.
    # Once claimed, subsequent calls return "" until mark_screenshot_done clears the flag.
    if pending_screenshot and not _screenshot_claimed:
        _screenshot_claimed = True
        log(f"check_screenshot_request: claiming screenshot for path={pending_screenshot}")
        return pending_screenshot
    return ""

def mark_screenshot_done():
    global pending_screenshot, screenshot_done_event, _screenshot_claimed
    pending_screenshot = None
    _screenshot_claimed = False
    screenshot_done_event.set()
    return True

class TrainingWorker(threading.Thread):
    def __init__(self, learner, config, experiment_manager=None):
        super().__init__(daemon=True)
        self.learner = learner
        self.config = config
        self.experiment_manager = experiment_manager
        self.batch_buffer = []
        self.running = True
        self.step_count = 0
        self.episode_last_floors = []
        self.episode_last_rewards = []
        self.last_known_mean_floor = None
        self.last_known_mean_reward = None
        self.lock = threading.Lock()

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

    def record_episode_end(self, floor, reward):
        with self.lock:
            self.episode_last_floors.append(floor)
            self.episode_last_rewards.append(reward)
            print(f"[Python] Recorded episode end at floor {floor}, reward {reward:.2f}. Count: {len(self.episode_last_floors)}")

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
        
        # Add mean last floor/reward if we have data
        with self.lock:
            if self.episode_last_floors:
                # Calculate mean of episodes that ended since last update
                self.last_known_mean_floor = sum(self.episode_last_floors) / len(self.episode_last_floors)
                self.last_known_mean_reward = sum(self.episode_last_rewards) / len(self.episode_last_rewards)
                self.episode_last_floors = [] # Clear for next update
                self.episode_last_rewards = []
            
            # Report the last known means to keep metrics visible in MLflow
            if self.last_known_mean_floor is not None:
                metrics['mean_last_floor'] = self.last_known_mean_floor
            if self.last_known_mean_reward is not None:
                metrics['mean_last_reward'] = self.last_known_mean_reward

        self.step_count += 1
        
        log_msg = f"[Python] Training Step {self.step_count}: Loss={metrics['loss']:.4f}, Policy Loss={metrics['policy_loss']:.4f}, Entropy Alpha={metrics['alpha']:.4f}"
        if 'mean_last_floor' in metrics:
            log_msg += f", Mean Last Floor={metrics['mean_last_floor']:.2f}"
        if 'mean_last_reward' in metrics:
            log_msg += f", Mean Last Reward={metrics['mean_last_reward']:.2f}"
        print(log_msg)
        
        if self.experiment_manager:
            self.experiment_manager.log_metrics(self.step_count, metrics)

        if self.step_count % 10 == 0:
            checkpoint_path = f"/home/ubuntu/src/R-NaD-StS2/R-NaD/checkpoints/checkpoint_{self.step_count}.pkl"
            if self.experiment_manager:
                # Use experiment-specific subdir if possible
                checkpoint_path = os.path.join(self.experiment_manager.checkpoint_dir, f"checkpoint_{self.step_count}.pkl")
            
            self.learner.save_checkpoint(checkpoint_path, self.step_count)
            print(f"[Python] Saved checkpoint to {checkpoint_path}")
            
            if self.experiment_manager:
                self.experiment_manager.log_checkpoint_artifact(self.step_count, checkpoint_path)

training_worker = None

def load_model(checkpoint_path=None):
    global learner, rng_key, training_worker, config
    do_deferred_imports()
    
    state_dim = 128
    num_actions = 50
    config = RNaDConfig(
        batch_size=8, 
        unroll_length=32, 
        model_type="transformer",
        hidden_size=256,
        num_blocks=4,
        num_heads=4,
        seq_len=8,
        accumulation_steps=1 # Can be changed to test
    )
    learner = RNaDLearner(state_dim, num_actions, config)
    rng_key = jax.random.PRNGKey(42)

    # Initialize ExperimentManager
    exp_manager = None
    try:
        exp_manager = ExperimentManager(experiment_name="R-NaD-StS2", log_checkpoints=True)
        exp_manager.log_params(config)
        print(f"[Python] MLflow initialized. Run ID: {exp_manager.run_id}")
    except Exception as e:
        print(f"[Python] Warning: Failed to initialize MLflow: {e}")
    
    # If no checkpoint_path is provided, try RNAD_CHECKPOINT env var then auto-detect
    if checkpoint_path is None:
        checkpoint_path = os.environ.get("RNAD_CHECKPOINT")
        if checkpoint_path:
            print(f"[Python] Using checkpoint from environment variable: {checkpoint_path}")

    if checkpoint_path is None:
        checkpoint_dir = "/home/ubuntu/src/R-NaD-StS2/R-NaD/checkpoints"
        if os.path.exists(checkpoint_dir):
            import glob
            import re
            
            # Search recursively for checkpoints because ExperimentManager saves them in subdirs
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "**", "checkpoint_*.pkl"), recursive=True)
            if checkpoints:
                # Extract step number and find the max
                def get_step(path):
                    match = re.search(r"checkpoint_(\d+)\.pkl", os.path.basename(path))
                    return int(match.group(1)) if match else -1
                
                latest_checkpoint = max(checkpoints, key=get_step)
                checkpoint_path = latest_checkpoint
                print(f"[Python] Auto-detected latest checkpoint: {checkpoint_path}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            step = learner.load_checkpoint(checkpoint_path)
            print(f"[Python] Loaded JAX model from {checkpoint_path} at step {step}")
        except Exception as e:
            print(f"[Python] Error loading checkpoint {checkpoint_path}: {e}")
            learner.init(rng_key)
            print("[Python] Falling back to new JAX model initialization")
    else:
        learner.init(rng_key)
        print("[Python] Initialized new JAX model")
    
    # Start training worker if learning behavior is expected
    if training_worker is None:
        training_worker = TrainingWorker(learner, config, experiment_manager=exp_manager)
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

def compute_reward(state, state_type=None):
    """Compute the reward for the current state.
    This is now a final reward: returns 0.0 unless the state is game_over.
    """
    if state_type != "game_over":
        return 0.0
    
    # Final reward: floor progression (normalized) + victory/defeat bonus
    floor = state.get("floor", 0)
    victory = state.get("victory", False)
    
    reward = floor / 50.0
    if victory:
        reward += 1.0
    else:
        reward -= 1.0
            
    return reward

def get_action_mask(state, masked_reward_indices=None):
    mask = np.zeros(50, dtype=bool)
    state_type = state.get("type", "unknown")
    
    if state_type == "combat":
        hand = state.get("hand", [])
        for i in range(min(len(hand), 10)):
            if hand[i].get("isPlayable"):
                mask[i] = True
        
        potions = state.get("potions", [])
        for i in range(min(len(potions), 5)):
            if potions[i].get("canUse"):
                mask[10 + i] = True
        
        mask[15] = True # End turn is almost always valid in play phase
    
    elif state_type == "rewards":
        rewards = state.get("rewards", [])
        for i in range(min(len(rewards), 10)):
            if masked_reward_indices and i in masked_reward_indices:
                continue
            mask[16 + i] = True
        if state.get("can_proceed"):
            mask[26] = True
            
    elif state_type == "map":
        next_nodes = state.get("next_nodes", [])
        if next_nodes:
            mask[27] = True # Simple mapping for now: 27 means "take first valid map node"
            
    elif state_type == "event":
        options = state.get("options", [])
        for i in range(min(len(options), 10)):
            if not options[i].get("is_locked"):
                mask[28] = True # Using 28 as a generic event action for now
                break
                
    elif state_type == "rest_site":
        options = state.get("options", [])
        for i in range(min(len(options), 5)):
            if options[i].get("is_enabled"):
                mask[29] = True
                break
        if state.get("can_proceed"):
            mask[26] = True

    # Action 49 (Wait) is a fallback
    if state_type in ["grid_selection", "hand_selection"]:
        # Ensure we always have a valid action in selection screens
        # Actions for grid/hand selection start from 26 (proceed/confirm) or higher
        mask[26] = True # Proceed/Confirm
        mask[49] = False # Avoid waiting in selection screens if possible
        
    if not np.any(mask):
        mask[49] = True
        
    return mask

def predict_action(state_json):
    global command_queue, learning_active, current_seed, current_trajectory, learner, rng_key
    
    try:
        state = json.loads(state_json)
        state_type = state.get("type", "unknown")
        
        # Track episode end for mean last floor calculation
        if state_type == "game_over":
            # Use a flag to record only once per game_over screen. 
            # Reset the flag when we see any gameplay state.
            if not getattr(predict_action, 'episode_end_recorded', False):
                floor = state.get("floor", 1)
                # Ensure we include the terminal reward in the logged metric
                terminal_reward = compute_reward(state, state_type)
                total_reward = getattr(predict_action, 'session_cumulative_reward', 0.0) + terminal_reward
                log(f"Episode end detected at floor {floor}, final reward {total_reward:.2f}. Recording...")
                if training_worker:
                    training_worker.record_episode_end(floor, total_reward)
                predict_action.episode_end_recorded = True
                # Reset reward for next episode (will be reset below too, but safe)
                predict_action.session_cumulative_reward = 0.0
            predict_action.skipped_reward_indices = set()
        elif state_type in ["combat", "map", "event", "rest_site", "shop", "treasure"]:
            predict_action.episode_end_recorded = False
            # Ensure cumulative reward is initialized
            if not hasattr(predict_action, 'session_cumulative_reward'):
                predict_action.session_cumulative_reward = 0.0
            # Reset skipped rewards when leaving rewards flow
            predict_action.skipped_reward_indices = set()
            predict_action.last_reward_floor = state.get("floor", -1)
        elif state_type == "rewards":
            predict_action.episode_end_recorded = False
            if not hasattr(predict_action, 'skipped_reward_indices') or getattr(predict_action, 'last_reward_floor', -1) != state.get("floor"):
                predict_action.skipped_reward_indices = set()
                predict_action.last_reward_floor = state.get("floor")
        elif state_type in ["main_menu", "none"]:
            predict_action.session_cumulative_reward = 0.0
            predict_action.episode_end_recorded = False
            predict_action.skipped_reward_indices = set()

        log(f"predict_action called. state_type: {state_type}, command_queue size: {command_queue.qsize()}")
        
        # Debug: write last state to a local file for monitoring
        try:
            last_state_path = os.path.join(LOG_DIR, "rnad_last_state.json")
            with open(last_state_path, "w") as f:
                f.write(state_json)
        except:
            pass

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
        
        # Calculate Action Mask
        masked_rewards = getattr(predict_action, 'skipped_reward_indices', set())
        mask = get_action_mask(state, masked_reward_indices=masked_rewards)
        mask_jnp = jnp.array(mask)
        
        # Inference
        logits, value = learner.network.apply(learner.params, None, state_vec[None, :])
        
        # Apply Masking
        # Set illegal actions to a very low value before softmax
        masked_logits = jnp.where(mask_jnp, logits, -1e9)
        
        probs = jax.nn.softmax(masked_logits, axis=-1)
        
        # Sample action from masked distribution
        rng_key, subkey = jax.random.split(rng_key)
        action_idx = jax.random.categorical(subkey, masked_logits).item()
        log_prob = jax.nn.log_softmax(masked_logits)[0, action_idx].item()
        
        # Log if the model's preferred (original top) action was masked
        original_top_action = jnp.argmax(logits).item()
        if not mask[original_top_action]:
             log(f"Model preferred action {original_top_action} but it was masked. Masked selected: {action_idx}")
        
        action = {"action": "wait"} # Default
        
        if action_idx == 49:
            pass # already wait
        
        elif state_type == "combat":
            hand = state.get("hand", [])
            potions = state.get("potions", [])
            
            if action_idx < 10 and action_idx < len(hand):
                card = hand[action_idx]
                action = {"action": "play_card", "card_id": card.get("id")}
            elif 10 <= action_idx < 15:
                potion_idx = action_idx - 10
                if potion_idx < len(potions):
                    action = {"action": "use_potion", "index": potion_idx}
            elif action_idx == 15:
                action = {"action": "end_turn"}
        
        elif state_type == "rewards":
            rewards = state.get("rewards", [])
            if 16 <= action_idx < 26:
                reward_idx = action_idx - 16
                if reward_idx < len(rewards):
                    action = {"action": "select_reward", "index": reward_idx}
            elif action_idx == 26:
                action = {"action": "proceed"}

        # Fallback to heuristic for complex types not fully mapped to action_idx yet
        if action["action"] == "wait" and state_type not in ["unknown"]:
             log(f"Mapping action_idx {action_idx} to heuristic for state: {state_type}")
             action = get_heuristic_action(state)

        # Trajectory collection
        if learning_active:
            reward = compute_reward(state, state_type)
            
            # Accumulate session reward
            if not hasattr(predict_action, 'session_cumulative_reward'):
                predict_action.session_cumulative_reward = 0.0
            predict_action.session_cumulative_reward += reward

            current_trajectory.append({
                "obs": state_vec,
                "act": int(action_idx),
                "rew": float(reward),
                "log_prob": float(log_prob)
            })
            
            if len(current_trajectory) >= config.unroll_length:
                experience_queue.put(list(current_trajectory))
                current_trajectory = []

        # If we chose to skip a card reward, remember this to mask it in next rewards screen call
        if action.get("action") == "click_reward_button":
             if action.get("index") is not None:
                 buttons = state.get("buttons", [])
                 if action["index"] < len(buttons):
                     btn_name = buttons[action["index"]].get("name", "").lower()
                     if "skip" in btn_name or "remov" in btn_name or "dismiss" in btn_name: 
                         last_reward_idx = getattr(predict_action, 'last_selected_reward_idx', None)
                         if last_reward_idx is not None:
                             log(f"Detected SKIP in card reward. Masking reward index {last_reward_idx} for floor {state.get('floor')}")
                             predict_action.skipped_reward_indices.add(last_reward_idx)

        if action.get("action") == "select_reward":
            predict_action.last_selected_reward_idx = action.get("index")

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
        if not hand:
            # If the hand is completely empty, we might be in the middle of a draw animation
            # Or recovering from a hand disruption. It's safer to wait just a bit.
            # But what if we actually have 0 cards? We have to ensure we don't infinite wait.
            # For this simple heuristic, let's look at player state. If we have no playable cards,
            # and hand is not empty, then end turn. If hand IS empty, wait.
            return {"action": "wait"}
            
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

    elif state_type == "none":
        global learning_active, current_seed
        if learning_active:
            cmd = f"start_game:{current_seed}" if current_seed else "start_game"
            return {"action": "command", "command": cmd}

    elif state_type == "event":
        options = [o for o in state.get("options", []) if not o.get("is_locked")]
        if options:
            return {"action": "select_event_option", "index": random.choice(options).get("index")}
            
    elif state_type == "rest_site":
        if state.get("can_proceed"):
            return {"action": "proceed"}
        options = [o for o in state.get("options", []) if o.get("is_enabled")]
        if options:
            return {"action": "select_rest_site_option", "index": random.choice(options).get("index")}
            
    elif state_type == "shop":
        # Just proceed for simplicity, or buy a random affordable item
        return {"action": "shop_proceed"}
        
    elif state_type == "treasure":
        if state.get("has_chest"):
            return {"action": "open_chest"}
        if state.get("can_proceed"):
            return {"action": "proceed"}
            
    elif state_type == "treasure_relics":
        relics = state.get("relics", [])
        if relics:
            return {"action": "select_treasure_relic", "index": relics[0].get("index")}
            
    elif state_type == "card_reward":
        buttons = state.get("buttons", [])
        if buttons:
            skip_btns = [b for b in buttons if b.get("name", "").lower() == "skip"]
            if skip_btns:
                return {"action": "click_reward_button", "index": skip_btns[0].get("index")}
            return {"action": "click_reward_button", "index": buttons[0].get("index")}

    elif state_type == "grid_selection":
        if state.get("is_confirming"):
            return {"action": "confirm_selection"}
        if state.get("can_skip"):
            return {"action": "select_grid_card", "index": -1}
        cards = state.get("cards", [])
        if cards:
            return {"action": "select_grid_card", "index": random.choice(cards).get("index")}
            
    elif state_type == "hand_selection":
        if state.get("is_confirming"):
            return {"action": "confirm_selection"}
        cards = state.get("cards", [])
        if cards:
            return {"action": "select_hand_card", "index": random.choice(cards).get("index")}
        
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
            global pending_screenshot, screenshot_done_event
            timestamp = int(time.time())
            screenshot_dir = os.path.join(LOG_DIR, "screenshots")
            if not os.path.exists(screenshot_dir):
                os.makedirs(screenshot_dir, exist_ok=True)
            path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
            pending_screenshot = path
            screenshot_done_event.clear()
            
            log(f"Screenshot requested at {path}. Waiting for game...")
            
            # Wait up to 30 seconds for the game's internal viewport capture
            completed = screenshot_done_event.wait(timeout=30.0)
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            if completed:
                log(f"Screenshot request fulfilled (path: {path}).")
                self.wfile.write(json.dumps({"status": "success", "path": path}).encode())
            else:
                log("Screenshot request timed out (30s).")
                # Clear the request if it timed out so we don't take it later
                pending_screenshot = None
                self.wfile.write(json.dumps({"status": "error", "message": "timeout"}).encode())
            
        elif parsed_path.path == "/new_game":
            query_components = parse_qs(parsed_path.query)
            seed = query_components.get("seed", [None])[0]
            if seed:
                current_seed = seed
                log(f"Updated current_seed to: {current_seed}")
            
            cmd = f"start_game:{current_seed}" if current_seed else "start_game"
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
