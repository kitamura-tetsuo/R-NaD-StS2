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
from event_dict import get_event_features
try:
    from PIL import ImageGrab
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Configure JAX to not preallocate all memory and force CPU if needed
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["JAX_PLATFORMS"] = "cpu"

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

# Card Vocabulary Mapping
CARD_VOCAB = {
    "UNKNOWN": 0,
    "STRIKE_IRONCLAD": 1,
    "DEFEND_IRONCLAD": 2,
    "BASH": 3,
    "ANGER": 4,
    "BODY_SLAM": 5,
    "CLASH": 6,
    "CLEAVE": 7,
    "CLOTHESLINE": 8,
    "FLEX": 9,
    "HAVOC": 10,
    "IRON_WAVE": 11,
    "PERFECTED_STRIKE": 12,
    "POMMEL_STRIKE": 13,
    "SHRUG_IT_OFF": 14,
    "SWORD_BOOMERANG": 15,
    "THUNDER_CLAP": 16,
    "TRUE_GRIT": 17,
    "TWIN_STRIKE": 18,
    "WARCRY": 19,
    "WILD_STRIKE": 20,
    "ARMAMENTS": 21,
    "BLOOD_FOR_BLOOD": 22,
    "BLOOD_LETTING": 23,
    "BURNING_BARRIER": 24, # StS2 specific?
    "CARNAGE": 25,
    "COMBUST": 26,
    "DARK_EMBRACE": 27,
    "DISARM": 28,
    "DUAL_WIELD": 29,
    "ENTRENCH": 30,
    "EVOLVE": 31,
    "FEEL_THE_BURN": 32,
    "FIRE_BREATHING": 33,
    "FLAME_BARRIER": 34,
    "GHOSTLY_ARMOR": 35,
    "HEMOKINESIS": 36,
    "INFERNAL_BLADE": 37,
    "INFLAME": 38,
    "INTIMIDATE": 39,
    "METALLICIZE": 40,
    "POWER_THROUGH": 41,
    "PUMMEL": 42,
    "RAGE": 43,
    "RAMPAGE": 44,
    "RECKLESS_CHARGE": 45,
    "RUPTURE": 46,
    "SEARING_BLOW": 47,
    "SECOND_WIND": 48,
    "SEEING_RED": 49,
    "SENTINEL": 50,
    "SEVER_SOUL": 51,
    "SHOCKWAVE": 52,
    "SPOT_WEAKNESS": 53,
    "UPPERCUT": 54,
    "WHIRLWIND": 55,
    "BARRICADE": 56,
    "BERSERK": 57,
    "BRUTALITY": 58,
    "CORRUPTION": 59,
    "DEMON_FORM": 60,
    "DOUBLE_TAP": 61,
    "EXHUME": 62,
    "FEED": 63,
    "FIEND_FIRE": 64,
    "HEAL": 65,
    "IMMOLATE": 66,
    "IMPERVIOUS": 67,
    "JUGGERNAUT": 68,
    "LIMIT_BREAK": 69,
    "OFFERING": 70,
    "REAPER": 71,
    "SLIMED": 72,
    "DAZED": 73,
    "VOID": 74,
    "BURN": 75,
    "WOUND": 76,
    "ASCENDERS_BANE": 77
}
VOCAB_SIZE = 100 # Fixed size to allow for new cards

RELIC_VOCAB = {
    "UNKNOWN": 0,
    "BURNING_BLOOD": 1,
    "RING_OF_THE_SNAKE": 2,
    "CRACKED_CORE": 3,
    "PURE_WATER": 4,
    "AKABEKO": 5,
    "ANCHOR": 6,
    "ANCIENT_TEA_SET": 7,
    "ART_OF_WAR": 8,
    "BAG_OF_MARBLES": 9,
    "BAG_OF_PREPARATION": 10,
    "BLOOD_VIAL": 11,
    "BRONZE_SCALES": 12,
    "CENTENNIAL_PUZZLE": 13,
    "CERAMIC_FISH": 14,
    "DREAM_CATCHER": 15,
    "HAPPY_FLOWER": 16,
    "LANTERN": 17,
    "MEAD_WHIP": 18,
    "NUNCHAKU": 19,
    "ODDLY_SMOOTH_STONE": 20,
    "ORICHALCUM": 21,
    "PEN_NIB": 22,
    "POTION_BELT": 23,
    "PRESERVED_INSECT": 24,
    "REGAL_PILLOW": 25,
    "SMILING_MASK": 26,
    "STRAW_DOLL": 27,
    "TOY_ORNITHOPTER": 28,
    "VAJRA": 29,
    "WAR_PAINT": 30,
    "WHETSTONE": 31
}
RELIC_VOCAB_SIZE = 50

POWER_VOCAB = {
    "UNKNOWN": 0,
    "STRENGTH": 1,
    "DEXTERITY": 2,
    "FOCUS": 3,
    "VULNERABLE": 4,
    "WEAK": 5,
    "FRAIL": 6,
    "NO_BLOCK_NEXT_TURN": 7,
    "ARTIFACT": 8,
    "THORNS": 9,
    "METALLICIZE": 10,
    "PLATED_ARMOR": 11,
    "REGEN": 12,
    "RITUAL": 13,
    "COMBUST": 14,
    "DARK_EMBRACE": 15,
    "EVOLVE": 16,
    "FEEL_THE_BURN": 17,
    "FIRE_BREATHING": 18,
    "FLAME_BARRIER": 19
}
POWER_VOCAB_SIZE = 20

BOSS_VOCAB = {
    "UNKNOWN": 0,
    "SLIME_BOSS": 1,
    "THE_GUARDIAN": 2,
    "HEXAGHOST": 3,
    "COLLECTOR": 4,
    "AUTOMATON": 5,
    "CHAMP": 6,
    "AWAKENED_ONE": 7,
    "TIME_EATER": 8,
    "DONU_DECA": 9,
    "THE_TWINS": 10,
    "THE_CONJURER": 11,
    "THE_GHOST": 12
}
BOSS_VOCAB_SIZE = 20

def get_boss_idx(boss_id):
    if not boss_id: return 0
    return BOSS_VOCAB.get(boss_id, 0)

def get_card_idx(card_id):
    if not card_id: return 0
    # Clean up (e.g., remove name suffixes if any)
    cid = card_id.split('+')[0].strip().upper()
    return CARD_VOCAB.get(cid, 0)

def get_relic_idx(relic_id):
    if not relic_id: return 0
    rid = relic_id.upper()
    return RELIC_VOCAB.get(rid, 0)

def get_power_idx(power_id):
    if not power_id: return 0
    pid = power_id.upper()
    return POWER_VOCAB.get(pid, 0)

def encode_bow(card_ids):
    do_deferred_imports()
    assert np is not None
    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
    if not card_ids: return vec
    for cid in card_ids:
        idx = get_card_idx(cid)
        if idx < VOCAB_SIZE:
            vec[idx] += 1.0
    return vec

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
        self.last_known_mean_floor: float | None = None
        self.last_known_mean_reward: float | None = None
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
        do_deferred_imports()
        assert np is not None
        assert jnp is not None
        
        # Transpose to (T, B, ...)
        # Trajectories might have different lengths, so we pad them
        max_len = self.config.unroll_length
        # obs is now a dictionary of lists of lists
        padded_obs_dict = {
            "global": [],
            "combat": [],
            "draw_bow": [],
            "discard_bow": [],
            "exhaust_bow": [],
            "master_bow": [],
            "map": [],
            "event": [],
            "state_type": []
        }
        padded_act = []
        padded_rew = []
        padded_mask = []
        padded_log_prob = []
        valid_mask = []

        for traj in self.batch_buffer:
            l = len(traj)
            
            # obs_traj is a list of dicts
            obs_traj = [t['obs'] for t in (traj if isinstance(traj, list) else [])]
            
            # Pad each element in the dict
            for key in padded_obs_dict.keys():
                val_traj = [o[key] for o in obs_traj]
                # Pad with zeros (or last for state_type if preferred, but 2 is fine as default)
                if key == "state_type":
                    val_traj += [np.int32(2)] * (max_len - l)
                else:
                    val_traj += [np.zeros_like(val_traj[0])] * (max_len - l)
                padded_obs_dict[key].append(val_traj)

            # Pad act
            act_traj = [t['act'] for t in traj]
            act_traj += [0] * (max_len - l)
            padded_act.append(act_traj)

            # Pad rew
            rew_traj = [t['rew'] for t in traj]
            rew_traj += [0.0] * (max_len - l)
            padded_rew.append(rew_traj)

            # Pad mask
            mask_traj = [t['mask'] for t in traj]
            mask_traj += [np.zeros_like(mask_traj[0])] * (max_len - l)
            padded_mask.append(mask_traj)

            # Pad log_prob
            lp_traj = [t['log_prob'] for t in traj]
            lp_traj += [0.0] * (max_len - l)
            padded_log_prob.append(lp_traj)

            # Valid mask
            v_mask = [1.0] * l + [0.0] * (max_len - l)
            valid_mask.append(v_mask)

        # Build JAX-ready obs dictionary
        # Each element should be (T, B, dim) - currently (B, T, dim)
        jax_obs = {
            k: jnp.array(np.array(v).transpose(1, 0, *range(2, np.array(v).ndim)))
            for k, v in padded_obs_dict.items()
        }
        
        act = np.array(padded_act)
        rew = np.array(padded_rew)
        mask = np.array(padded_mask)
        log_prob = np.array(padded_log_prob)
        valid = np.array(valid_mask)

        batch = {
            'obs': jax_obs,
            'act': jnp.array(act.transpose(1, 0)),
            'rew': jnp.array(rew.transpose(1, 0)),
            'mask': jnp.array(mask.transpose(1, 0, 2)),
            'log_prob': jnp.array(log_prob.transpose(1, 0)),
            'valid': jnp.array(valid.transpose(1, 0))
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
    
    num_actions = 100
    config = RNaDConfig(
        batch_size=8, 
        unroll_length=256, 
        model_type="transformer",
        hidden_size=512,
        num_blocks=8,
        num_heads=8,
        seq_len=16,
        accumulation_steps=2 # Can be changed to test
    )
    # Updated dummy state for structured dictionary input
    dummy_obs = {
        "global": jnp.zeros((1, 128)),
        "combat": jnp.zeros((1, 384)),
        "map": jnp.zeros((1, 2048)),
        "event": jnp.zeros((1, 128)),
        "state_type": jnp.zeros((1,), dtype=jnp.int32)
    }
    do_deferred_imports()
    assert RNaDLearner is not None
    assert jax is not None
    assert ExperimentManager is not None
    
    learner = RNaDLearner(None, num_actions, config) # state_dim is now unused/ignored in init
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
    do_deferred_imports()
    assert np is not None
    """Encodes the game state into a structured dictionary of NumPy arrays."""
    state_type = state.get("type", "unknown")
    
    # State type mapping
    # 0: Combat
    # 1: Map
    # 2: Event-like (Rewards, Event, Shop, Rest, Treasure, etc.)
    type_map = {
        "combat": 0,
        "map": 1,
        "rewards": 2,
        "event": 2,
        "rest_site": 2,
        "shop": 2,
        "treasure": 2,
        "game_over": 2,
        "treasure_relics": 2,
        "card_reward": 2,
        "grid_selection": 3,
        "hand_selection": 4
    }
    st_idx = type_map.get(state_type, 2)
    
    # --- Global Features (Size 128) ---
    global_vec = np.zeros(128, dtype=np.float32)
    global_vec[0] = state.get("floor", 0) / 50.0
    global_vec[1] = state.get("gold", 0) / 500.0
    
    player = state.get("player", {})
    global_vec[2] = player.get("hp", 0) / 100.0
    global_vec[3] = player.get("maxHp", 100) / 100.0
    global_vec[4] = player.get("block", 0) / 50.0
    global_vec[5] = player.get("energy", 0) / 5.0
    global_vec[6] = player.get("stars", 0) / 10.0
    
    # Potion presence
    potions = state.get("potions", [])
    for i in range(min(len(potions), 5)):
        if potions[i].get("id") != "empty":
            global_vec[10 + i] = 1.0
            
    # Boss ID
    boss_id = state.get("boss", "Unknown")
    global_vec[20] = get_boss_idx(boss_id) / float(BOSS_VOCAB_SIZE)
    
    # Relics (Multi-hot, size 50, starting at index 30)
    relics = state.get("relics", [])
    if not relics and player: # Fallback for combat state where player is a child
        relics = player.get("relics", [])
    for rid in relics:
        idx = get_relic_idx(rid)
        if 0 < idx < RELIC_VOCAB_SIZE:
            global_vec[30 + idx] = 1.0

    # --- Combat Features (Size 384) ---
    combat_vec = np.zeros(384, dtype=np.float32)
    draw_bow = np.zeros(VOCAB_SIZE, dtype=np.float32)
    discard_bow = np.zeros(VOCAB_SIZE, dtype=np.float32)
    exhaust_bow = np.zeros(VOCAB_SIZE, dtype=np.float32)
    master_bow = np.zeros(VOCAB_SIZE, dtype=np.float32)

    if st_idx == 0:
        # Piles
        draw_bow = encode_bow(player.get("drawPile", []))
        discard_bow = encode_bow(player.get("discardPile", []))
        exhaust_bow = encode_bow(player.get("exhaustPile", []))
        master_bow = encode_bow(player.get("masterDeck", []))

        # Pile counts for legacy/redundancy
        combat_vec[0] = len(player.get("drawPile", [])) / 30.0
        combat_vec[1] = len(player.get("discardPile", [])) / 30.0
        combat_vec[2] = len(player.get("exhaustPile", [])) / 30.0
        
        # Hand cards (up to 10 cards, 10 features each)
        hand = state.get("hand", [])
        for i in range(min(len(hand), 10)):
            card = hand[i]
            base_idx = 10 + i * 10
            # Card ID index for embedding
            combat_vec[base_idx] = get_card_idx(card.get("id", ""))
            combat_vec[base_idx + 1] = 1.0 if card.get("isPlayable") else 0.0
            
            target_type = card.get("targetType", "None")
            tt_map = {"SingleEnemy": 1, "AllEnemy": 2, "RandomEnemy": 3, "None": 0, "Self": 4}
            combat_vec[base_idx + 2] = tt_map.get(target_type, 0) / 10.0
            combat_vec[base_idx + 3] = card.get("cost", 0) / 5.0
            combat_vec[base_idx + 4] = card.get("baseDamage", 0) / 20.0
            combat_vec[base_idx + 5] = card.get("baseBlock", 0) / 20.0
            combat_vec[base_idx + 6] = card.get("magicNumber", 0) / 10.0
            combat_vec[base_idx + 7] = 1.0 if card.get("upgraded") else 0.0
            combat_vec[base_idx + 8] = card.get("currentDamage", 0) / 50.0
            combat_vec[base_idx + 9] = card.get("currentBlock", 0) / 50.0

        # Enemies (up to 5 enemies, 12 features each)
        # Offset to 110 to avoid overlap with hand cards
        enemies = state.get("enemies", [])
        for i in range(min(len(enemies), 5)):
            enemy = enemies[i]
            base_idx = 110 + i * 12
            combat_vec[base_idx] = 1.0 # Alive
            combat_vec[base_idx + 1] = enemy.get("hp", 0) / 200.0
            combat_vec[base_idx + 2] = enemy.get("maxHp", 1) / 200.0
            combat_vec[base_idx + 3] = enemy.get("block", 0) / 50.0
            
            intents = enemy.get("intents", [])
            for j in range(min(len(intents), 2)):
                intent = intents[j]
                intent_idx = base_idx + 4 + j * 4
                it_map = {"Attack": 1, "Defense": 2, "AttackDefense": 3, "Buff": 4, "Debuff": 5, "StrongDebuff": 6, "Stun": 7}
                combat_vec[intent_idx] = it_map.get(intent.get("type"), 0) / 10.0
                combat_vec[intent_idx + 1] = intent.get("damage", 0) / 50.0
                combat_vec[intent_idx + 2] = intent.get("repeats", 1) / 5.0
                
        # Powers (starting at 200)
        # Player powers (up to 10, index 200-219)
        p_powers = player.get("powers", [])
        for i in range(min(len(p_powers), 10)):
            p = p_powers[i]
            base_idx = 200 + i * 2
            combat_vec[base_idx] = get_power_idx(p.get("id")) / float(POWER_VOCAB_SIZE)
            combat_vec[base_idx + 1] = p.get("amount", 0) / 10.0
            
        # Enemy powers (up to 5 enemies, 10 powers each, starting at 220)
        for i in range(min(len(enemies), 5)):
            e_powers = enemies[i].get("powers", [])
            enemy_base = 220 + i * 20
            for j in range(min(len(e_powers), 10)):
                p = e_powers[j]
                idx = enemy_base + j * 2
                combat_vec[idx] = get_power_idx(p.get("id")) / float(POWER_VOCAB_SIZE)
                combat_vec[idx + 1] = p.get("amount", 0) / 10.0
                
    # --- Map Features (Size 2048) ---
    map_vec = np.zeros(2048, dtype=np.float32)
    if st_idx == 1:
        nodes = state.get("nodes", [])
        current_pos = state.get("current_pos", {})
        
        # Encode up to 256 nodes, 8 features each
        # [presence, row, col, type, is_current]
        for i in range(min(len(nodes), 256)):
            node = nodes[i]
            base_idx = i * 8
            row = node.get("row", 0)
            col = node.get("col", 0)
            
            map_vec[base_idx] = 1.0
            map_vec[base_idx + 1] = row / 20.0
            map_vec[base_idx + 2] = col / 7.0
            
            nt_map = {"Monster": 1, "Elite": 2, "Event": 3, "Rest": 4, "Shop": 5, "Treasure": 6, "Boss": 7}
            map_vec[base_idx + 3] = nt_map.get(node.get("type"), 0) / 10.0
            
            if current_pos and row == current_pos.get("row") and col == current_pos.get("col"):
                map_vec[base_idx + 4] = 1.0

    # --- Event Features (Size 128) ---
    event_vec = np.zeros(128, dtype=np.float32)
    if st_idx == 2:
        if state_type == "rewards":
            rewards = state.get("rewards", [])
            for i in range(min(len(rewards), 10)):
                reward = rewards[i]
                base_idx = i * 4
                event_vec[base_idx] = 1.0
                rt_map = {"Gold": 1, "Card": 2, "Relic": 3, "Potion": 4, "Curse": 5}
                event_vec[base_idx + 1] = rt_map.get(reward.get("type"), 0) / 10.0
        elif state_type == "event":
            options = state.get("options", [])
            event_id = state.get("id", "Unknown")
            for i in range(min(len(options), 10)):
                # Base features: presence and locked status
                event_vec[i] = 1.0 if not options[i].get("is_locked") else 0.5
                
                # Rich features from dictionary (10 features per option, offset by 20)
                rich_feats = get_event_features(event_id, i)
                base_idx = 20 + i * 10
                event_vec[base_idx : base_idx + 10] = rich_feats
        elif state_type == "shop":
             # Placeholder for shop
             event_vec[0] = 1.0
        elif state_type in ["grid_selection", "hand_selection"]:
            # Feature encoding for card selection
            cards = state.get("cards", [])
            for i in range(min(len(cards), 10)):
                card = cards[i]
                base_idx = i * 4
                event_vec[base_idx] = 1.0 # Presence flag
                
                # Card ID index
                event_vec[base_idx + 1] = get_card_idx(card.get("id", ""))
                
                event_vec[base_idx + 2] = 1.0 if card.get("upgraded") else 0.0
                event_vec[base_idx + 3] = card.get("cost", 0) / 5.0
    
    # Differentiation flag: 1.0 for permanent grid, -1.0 for temporary hand
    if state_type == "grid_selection":
        event_vec[90] = 1.0
    elif state_type == "hand_selection":
        event_vec[90] = -1.0

    return {
        "global": global_vec,
        "combat": combat_vec,
        "draw_bow": draw_bow,
        "discard_bow": discard_bow,
        "exhaust_bow": exhaust_bow,
        "master_bow": master_bow,
        "map": map_vec,
        "event": event_vec,
        "state_type": np.int32(st_idx)
    }

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
    do_deferred_imports()
    assert np is not None
    assert jnp is not None
    mask = np.zeros(100, dtype=bool)
    state_type = state.get("type", "unknown")
    
    if state_type == "combat":
        hand = state.get("hand", [])
        enemies = [e for e in state.get("enemies", []) if e.get("hp", 0) > 0]
        num_enemies = len(enemies)
        
        # 0-49: Cards (up to 10 cards * 5 targets)
        for i in range(min(len(hand), 10)):
            card = hand[i]
            if card.get("isPlayable"):
                target_type = card.get("targetType", "None")
                needs_target = "Enemy" in target_type or "Single" in target_type
                
                if needs_target:
                    for t in range(min(num_enemies, 5)):
                        mask[i * 5 + t] = True
                else:
                    # Self or No target cards use target_idx 0
                    mask[i * 5] = True
        
        # 50-74: Potions (up to 5 potions * 5 targets)
        potions = state.get("potions", [])
        for i in range(min(len(potions), 5)):
            potion = potions[i]
            if potion.get("canUse"):
                target_type = potion.get("targetType", "None")
                needs_target = "Enemy" in target_type or "Single" in target_type
                
                if needs_target:
                    for t in range(min(num_enemies, 5)):
                        mask[50 + i * 5 + t] = True
                else:
                    mask[50 + i * 5] = True
        
        # 75: End Turn
        mask[75] = True
        
        # 86: Proceed (Victory Bag)
        if state.get("can_proceed"):
            mask[86] = True
    
    elif state_type == "rewards":
        rewards = state.get("rewards", [])
        has_open_potion_slots = state.get("has_open_potion_slots", True)
        for i in range(min(len(rewards), 10)):
            if masked_reward_indices and i in masked_reward_indices:
                continue
            
            # Mask potion acquisition if slots are full
            reward = rewards[i]
            if reward.get("type") == "Potion" and not has_open_potion_slots:
                continue

            mask[76 + i] = True
        if state.get("can_proceed"):
            mask[86] = True # Proceed
            
    elif state_type == "map":
        next_nodes = state.get("next_nodes", [])
        if next_nodes:
            mask[87] = True # Map Node heuristic
            
    elif state_type == "event":
        options = state.get("options", [])
        for i in range(min(len(options), 10)):
            if not options[i].get("is_locked"):
                mask[88] = True # Event Option
                break
                
    elif state_type == "rest_site":
        options = state.get("options", [])
        for i in range(min(len(options), 5)):
            if options[i].get("is_enabled"):
                mask[89] = True # Rest Site Option
                break
        if state.get("can_proceed"):
            mask[86] = True # Proceed

    elif state_type in ["grid_selection", "hand_selection"]:
        mask[90] = True # Confirm/Interact
        
    elif state_type == "shop":
        mask[86] = True # Shop Proceed (heuristic for now)
        
    elif state_type == "treasure":
        if state.get("has_chest"):
            mask[91] = True # Open Chest
        if state.get("can_proceed"):
            mask[86] = True # Proceed
            
    elif state_type == "treasure_relics":
        mask[92] = True # Select Treasure Relic (heuristic)
        
    elif state_type == "card_reward":
        mask[93] = True # Card Reward interaction (heuristic)

    # 94-98: Discard Potion (indices 0-4)
    # This should be available in combat or potentially other states where potions are visible
    potions = state.get("potions", [])
    for i in range(min(len(potions), 5)):
        if potions[i].get("id") != "empty":
            mask[94 + i] = True

    # Action 99 (Wait) is disabled since waiting is handled in C#
    mask[99] = False
    
    if not np.any(mask):
        # If no actions are possible, we should ideally not be here due to C# busy check.
        # But if we are, we must ensure the model doesn't crash.
        # Since we want to AVOID wait, let's log this situation.
        log(f"WARNING: No valid actions in mask for state {state_type}. Current mask: {mask}")
        
    return mask

def predict_action(state_json):
    do_deferred_imports()
    assert np is not None
    assert jnp is not None
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
                
                # Flush trajectory if it exists
                if current_trajectory:
                    log(f"Flushing terminal trajectory of length {len(current_trajectory)}")
                    experience_queue.put(list(current_trajectory))
                    current_trajectory = []
                
                # Reset reward for next episode (will be reset below too, but safe)
                predict_action.session_cumulative_reward = 0.0
            predict_action.skipped_reward_indices = set()
            predict_action.last_processed_floor = -1
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
            predict_action.last_processed_floor = -1

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
        state_dict = encode_state(state)
        
        # Calculate Action Mask
        masked_rewards = getattr(predict_action, 'skipped_reward_indices', set())
        mask = get_action_mask(state, masked_reward_indices=masked_rewards)
        do_deferred_imports()
        assert np is not None
        assert jnp is not None
        mask_jnp = jnp.array(mask)
        
        # Prepare dictionary with leading batch dimensions for inference
        # state_dict has elements like (dim,) or (), need (1, dim) or (1,)
        batched_state = {
            k: jnp.array(v)[None, ...] for k, v in state_dict.items()
        }
        
        # Inference
        # Now passing batched_state dictionary to the network
        logits, value = learner.network.apply(learner.params, None, batched_state, mask[None, :].astype(jnp.float32))
        
        # Probs are calculated from logits which are already masked in the network
        probs = jax.nn.softmax(logits, axis=-1)
        
        # Sample action from masked distribution
        rng_key, subkey = jax.random.split(rng_key)
        action_idx = jax.random.categorical(subkey, logits).item()
        log_prob = jax.nn.log_softmax(logits)[0, action_idx].item()
        
        # Log if the model's preferred (original top) action was masked.
        # Note: Since the network now applies masking, we'd need to check raw logits before masking
        # to see what it "originally" wanted, but the network internally hides that.
        original_top_action = jnp.argmax(logits).item()
        
        action = {"action": "wait"} # Default
        
        if action_idx == 99:
            pass # already wait
        
        elif state_type == "combat":
            hand = state.get("hand", [])
            potions = state.get("potions", [])
            
            if action_idx < 50:
                card_idx = action_idx // 5
                target_idx = action_idx % 5
                if card_idx < len(hand):
                    card = hand[card_idx]
                    action = {"action": "play_card", "card_id": card.get("id"), "target_index": target_idx}
            elif 50 <= action_idx < 75:
                potion_linear_idx = action_idx - 50
                potion_idx = potion_linear_idx // 5
                target_idx = potion_linear_idx % 5
                if potion_idx < len(potions):
                    action = {"action": "use_potion", "index": potion_idx, "target_index": target_idx}
            elif action_idx == 75:
                action = {"action": "end_turn"}
        
        elif state_type == "rewards":
            rewards = state.get("rewards", [])
            if 76 <= action_idx < 86:
                reward_idx = action_idx - 76
                if reward_idx < len(rewards):
                    action = {"action": "select_reward", "index": reward_idx}
            elif action_idx == 86:
                action = {"action": "proceed"}
        
        elif action_idx == 86: # Dual mapping for proceed
             action = {"action": "proceed"}
        
        elif 94 <= action_idx < 99:
            potion_idx = action_idx - 94
            action = {"action": "discard_potion", "index": potion_idx}

        # Fallback to heuristic for complex types not fully mapped to action_idx yet
        if action["action"] == "wait" and state_type not in ["unknown"]:
             log(f"Mapping action_idx {action_idx} to heuristic for state: {state_type}")
             action = get_heuristic_action(state)

        # Trajectory collection
        if learning_active:
            # ▼追加: waitアクション(99)の場合は経験として蓄積しない
            if action_idx != 99:
                base_reward = compute_reward(state, state_type)
                
                # Intermediate reward for floor progression
                intermediate_reward = 0.0
                current_floor = state.get("floor", 0)
                last_processed_floor = getattr(predict_action, 'last_processed_floor', -1)
                
                if current_floor > last_processed_floor and last_processed_floor != -1:
                    # Calculate HP * 0.0000001
                    hp = 0
                    if "player" in state:
                        hp = state["player"].get("hp", 0)
                    elif "hp" in state: # Some state types might have HP at top level
                        hp = state.get("hp", 0)
                    
                    intermediate_reward = hp * 0.0000001
                    log(f"Intermediate reward for floor {current_floor}: {intermediate_reward:.10f} (HP: {hp})")
                    predict_action.last_processed_floor = current_floor
                elif last_processed_floor == -1 and current_floor > 0:
                    # Initialize last_processed_floor at the start of the game
                    predict_action.last_processed_floor = current_floor

                reward = base_reward + intermediate_reward
                
                # Accumulate session reward
                if not hasattr(predict_action, 'session_cumulative_reward'):
                    predict_action.session_cumulative_reward = 0.0
                predict_action.session_cumulative_reward += reward

                current_trajectory.append({
                    "obs": state_dict,
                    "act": int(action_idx),
                    "rew": float(reward),
                    "mask": mask.astype(np.float32),
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
        if state.get("can_proceed"):
            return {"action": "proceed"}
            
        hand = state.get("hand", [])
        if not hand:
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
        global learning_active, command_queue, current_seed
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
