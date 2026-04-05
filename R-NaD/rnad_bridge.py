import os
import sys
import time
import json
import datetime
import glob
import shutil
import filecmp
import importlib
import importlib.util

# Set XLA environment variables before JAX/other imports to prevent GPU OOM
# Godot game and JAX both need GPU memory.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5" # Allow game more room
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer=" # As suggested by JAX error

# Ensure stdout/stderr are unbuffered and also log to a file
import io
BRIDGE_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD"
LOG_DIR = os.path.join(BRIDGE_DIR, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "rnad_bridge.log")
MAX_LOG_SIZE = 10 * 1024 * 1024 # 10MB
BACKUP_COUNT = 3

DISCREPANCY_LOG_DIR = "/home/ubuntu/src/R-NaD-StS2/battle_simulator/discrepancy_logs"
if not os.path.exists(DISCREPANCY_LOG_DIR):
    os.makedirs(DISCREPANCY_LOG_DIR, exist_ok=True)

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

# Dedicated decision logger
DECISION_LOG_FILE = os.path.join(LOG_DIR, "rnad_decisions.log")
decision_logger = Logger(None, DECISION_LOG_FILE)

def log_decision(msg):
    decision_logger.write(f"{time.ctime()}: {msg}\n")

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
    # Insert the virtualenv right after the local directory so it takes precedence over system packages
    sys.path.insert(1, VENV_PATH)
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

try:
    import battle_simulator
    log("[Python] Successfully imported battle_simulator")
except Exception as e:
    log(f"[Python] Error importing battle_simulator: {e}")
    battle_simulator = None

import json

# Cards that should skip validation due to inherent randomness that the simulator doesn't handle
VALIDATION_SKIP_RANDOM = {"SWORD_BOOMERANG", "FIEND_FIRE", "HAVOC", "INFERNAL_BLADE"}
VALIDATION_SKIP_DRAW = {"POMMEL_STRIKE", "SHRUG_IT_OFF", "WARCRY", "OFFERING", "BATTLE_TRANCE", 
                        "BURNING_PACT", "HAVOC", "DARK_EMBRACE", "EVOLVE"}

class CombatValidator:
    def __init__(self):
        self.last_state = None
        self.last_action_idx = None
        self.enabled = True

    def to_simulator_json(self, cs_state):
        """Convert C# JSON state to Rust Simulator JSON format."""
        player_data = cs_state.get("player", {})
        
        def convert_powers(powers):
            res = []
            if not powers:
                return res
            for p in powers:
                p_id = p.get("id", "")
                # Map common powers, others will be Unknown
                res.append({
                    "id": p_id,
                    "amount": p.get("amount", 0)
                })
            return res

        def convert_creature(c, is_player=False):
            return {
                "id": c.get("id", "Player" if is_player else "Enemy"),
                "max_hp": c.get("maxHp", 100),
                "cur_hp": c.get("hp", 100),
                "block": c.get("block", 0),
                "is_minion": c.get("isMinion", False),
                "powers": convert_powers(c.get("powers", [])),
                "intents": c.get("intents", [])
            }

        def convert_card(c):
            if isinstance(c, str):
                c = {"id": c}
            # TargetType mapping
            tt_map = {
                "AnyEnemy": "Single", "SingleEnemy": "Single", 
                "AllEnemies": "All", "AllEnemy": "All", 
                "Self": "SelfTarget", "None": "None"
            }
            return {
                "id": c.get("id", "Unknown"),
                "cost": c.get("cost", 0),
                "base_damage": c.get("baseDamage", 0),
                "base_block": c.get("baseBlock", 0),
                "currentDamage": c.get("currentDamage", 0),
                "currentBlock": c.get("currentBlock", 0),
                "magic_number": c.get("magicNumber", 0),
                "target": tt_map.get(c.get("targetType", "None"), "None"),
                "is_upgraded": c.get("upgraded", False),
                "isPlayable": c.get("isPlayable", True)
            }

        sim_state = {
            "player": convert_creature(player_data, is_player=True),
            "enemies": [convert_creature(e) for e in cs_state.get("enemies", [])],
            "hand": [convert_card(c) for c in cs_state.get("hand", [])],
            "draw_pile": [convert_card(c) for c in cs_state.get("drawPile", [])],
            "discard_pile": [convert_card(c) for c in cs_state.get("discardPile", [])],
            "exhaust_pile": [convert_card(c) for c in cs_state.get("exhaustPile", [])],
            "potions": [
                {
                    "id": p.get("id", "empty"),
                    "name": p.get("name", "Empty Slot"),
                    "can_use": p.get("canUse", False),
                    "targetType": p.get("targetType", "None")
                } for p in cs_state.get("potions", [])
            ],
            "energy": player_data.get("energy", 0),
            "max_energy": player_data.get("maxEnergy", 0),
            "stars": player_data.get("stars", 0),
            "retains_block": cs_state.get("retains_block", False),
            "floor": cs_state.get("floor", 1)
        }
        return json.dumps(sim_state)

    def validate(self, current_state_json):
        if not self.enabled or self.last_state is None or self.last_action_idx is None:
            return

        # Pop the state to ensure we only validate it once per action.
        # This prevents duplicate logs from concurrent predict_action calls (due to polling).
        last_state = self.last_state
        last_action_idx = self.last_action_idx
        self.last_state = None
        self.last_action_idx = None

        try:
            current_state = json.loads(current_state_json)
            if current_state.get("type") != "combat":
                return

            # Simulate
            sim_json = self.to_simulator_json(last_state)
            sim = battle_simulator.Simulator.from_json(sim_json)
            
            # Action mapping: 0-49 cards, 75 end turn
            if last_action_idx < 50:
                card_idx = last_action_idx // 5
                target_idx = last_action_idx % 5
                # Verify if card exists
                if card_idx < len(last_state.get("hand", [])):
                    card = last_state["hand"][card_idx]
                    t_val = target_idx if needs_target(card) else None
                    sim.play_card(card_idx, t_val)
            else:
                return # Not a combat action we simulate yet

            # Get simulated outcome
            sim_outcome = json.loads(sim.get_state_json())
            
            # Compare important fields
            discrepancies = []
            
            # Player HP/Block
            p_real = current_state.get("player", {})
            p_sim = sim_outcome["player"]
            if p_real.get("hp") != p_sim["cur_hp"]:
                discrepancies.append(f"Player HP: real={p_real.get('hp')}, sim={p_sim['cur_hp']}")
            if p_real.get("block") != p_sim["block"]:
                discrepancies.append(f"Player Block: real={p_real.get('block')}, sim={p_sim['block']}")
            if p_real.get("energy") != sim_outcome.get("energy"):
                discrepancies.append(f"Player Energy: real={p_real.get('energy')}, sim={sim_outcome.get('energy')}")
            
            # Enemies HP/Block
            e_real = current_state.get("enemies", [])
            e_sim = sim_outcome["enemies"]
            for i in range(min(len(e_real), len(e_sim))):
                if e_real[i].get("hp") != e_sim[i]["cur_hp"]:
                    discrepancies.append(f"Enemy {i} HP: real={e_real[i].get('hp')}, sim={e_sim[i]['cur_hp']}")
                if e_real[i].get("block") != e_sim[i]["block"]:
                    discrepancies.append(f"Enemy {i} Block: real={e_real[i].get('block')}, sim={e_sim[i]['block']}")

            if discrepancies:
                # Filter discrepancies based on randomness rules
                filtered_discrepancies = []
                card_id = "UNKNOWN"
                if last_action_idx < 50:
                    card_idx = last_action_idx // 5
                    hand = last_state.get("hand", [])
                    if card_idx < len(hand):
                        card_id = hand[card_idx].get("id", "UNKNOWN").upper().split('+')[0].strip()
                
                is_random_target = card_id in VALIDATION_SKIP_RANDOM and len(last_state.get("enemies", [])) > 1
                is_draw = card_id in VALIDATION_SKIP_DRAW

                for d in discrepancies:
                    # Rule 1: Random Target Damage with multiple enemies -> Ignore Enemy HP/Block changes
                    if is_random_target and ("Enemy" in d):
                        continue
                    
                    # Rule 2: Card Draws -> Ignore HP/Block changes from unpredictable draw triggers (e.g. Fire Breathing/Evolve)
                    # Note: We still validate Energy as it should be stable for basic draws.
                    if is_draw and ("HP" in d or "Block" in d):
                        continue
                    
                    filtered_discrepancies.append(d)

                if not filtered_discrepancies:
                    log(f"[SIMULATOR VALIDATION] SUCCESS (Filtered volatile elements for {card_id})")
                    return

                log(f"[SIMULATOR VALIDATION] DISCREPANCY FOUND after action {last_action_idx} ({card_id}):")
                for d in filtered_discrepancies:
                    log(f"  - {d}")
                # Log full states for debugging
                log(f"  Sim State: {json.dumps(sim_outcome)}")
                # log(f"  Real State: {current_state_json}")

                # Save discrepancy logs
                try:
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    
                    # 1. State Before
                    with open(os.path.join(DISCREPANCY_LOG_DIR, f"state_before_{timestamp}.json"), "w") as f:
                        json.dump(last_state, f, indent=2)
                    
                    # 2. State After
                    with open(os.path.join(DISCREPANCY_LOG_DIR, f"state_after_{timestamp}.json"), "w") as f:
                        json.dump(current_state, f, indent=2)
                    
                    # 3. Action
                    action_info = {"action_idx": last_action_idx, "discrepancies": discrepancies}
                    if last_action_idx < 50:
                        card_idx = last_action_idx // 5
                        target_idx = last_action_idx % 5
                        hand = last_state.get("hand", [])
                        if card_idx < len(hand):
                            action_info["card"] = hand[card_idx].get("id")
                            action_info["target"] = target_idx
                    elif last_action_idx == 75:
                        action_info["action"] = "end_turn"
                    
                    with open(os.path.join(DISCREPANCY_LOG_DIR, f"action_{timestamp}.json"), "w") as f:
                        json.dump(action_info, f, indent=2)
                    
                    # 4. Summary Log
                    summary_log_path = os.path.join(DISCREPANCY_LOG_DIR, f"discrepancy_{timestamp}.log")
                    with open(summary_log_path, "w") as f:
                        f.write(f"[SIMULATOR VALIDATION] DISCREPANCY FOUND after action {last_action_idx} ({card_id}):\n")
                        for d in filtered_discrepancies:
                            f.write(f"  - {d}\n")
                        f.write(f"\nRelated Files:\n")
                        f.write(f"  - Action JSON: {os.path.join(DISCREPANCY_LOG_DIR, f'action_{timestamp}.json')}\n")
                        f.write(f"  - State Before: {os.path.join(DISCREPANCY_LOG_DIR, f'state_before_{timestamp}.json')}\n")
                        f.write(f"  - State After: {os.path.join(DISCREPANCY_LOG_DIR, f'state_after_{timestamp}.json')}\n")
                        f.write(f"  - Screenshot: {os.path.join(DISCREPANCY_LOG_DIR, f'screenshot_{timestamp}.png')}\n")
                    
                    # 5. Screenshot
                    global pending_screenshot, screenshot_done_event, _screenshot_claimed
                    screenshot_path = os.path.join(DISCREPANCY_LOG_DIR, f"screenshot_{timestamp}.png")
                    pending_screenshot = screenshot_path
                    _screenshot_claimed = False
                    screenshot_done_event.clear()
                    log(f"  [DISCREPANCY] Requesting screenshot: {screenshot_path}")
                    
                    # We don't block the main loop here to avoid freezing the game too long.
                    # The Godot mod will pick up the screenshot request in its next poll.
                    # screenshot_done_event.wait(timeout=5.0)
                    
                except Exception as log_e:
                    log(f"  [DISCREPANCY LOG ERROR] {log_e}")
            else:
                log(f"[SIMULATOR VALIDATION] SUCCESS after action {last_action_idx}")

        except Exception as e:
            log(f"[SIMULATOR VALIDATION] ERROR during validation: {e}")
        finally:
            pass

validator = CombatValidator()

def reload_battle_simulator(v_name=None):
    """Dynamically reload the battle_simulator module."""
    global battle_simulator, validator
    import importlib.util
    
    try:
        if v_name:
            target_so = os.path.join(BRIDGE_DIR, f"{v_name}.so")
        else:
            # Look for the latest battle_simulator_*.so
            sos = glob.glob(os.path.join(BRIDGE_DIR, "battle_simulator_*.so"))
            if not sos:
                log("[RELOAD] No versioned simulator found, trying default battle_simulator.so")
                target_so = os.path.join(BRIDGE_DIR, "battle_simulator.so")
            else:
                sos.sort(key=os.path.getmtime)
                target_so = sos[-1]
        
        if not os.path.exists(target_so):
            return f"Error: {target_so} not found"

        log(f"[RELOAD] Loading simulator from: {target_so}")
        
        # Unique module name to bypass sys.modules cache
        mod_name = f"battle_simulator_v{int(os.path.getmtime(target_so))}"
        
        spec = importlib.util.spec_from_file_location("battle_simulator", target_so)
        if spec is None:
            return "Error: Could not create spec"
            
        new_mod = importlib.util.module_from_spec(spec)
        # We also put it into sys.modules as 'battle_simulator' so future 'import battle_simulator' works
        sys.modules["battle_simulator"] = new_mod
        spec.loader.exec_module(new_mod)
        
        battle_simulator = new_mod
        validator = CombatValidator() # Re-init validator with new module
        
        log(f"[RELOAD] Successfully reloaded battle_simulator as {mod_name}")
        return "success"
    except Exception as e:
        err = f"Error during reload: {e}"
        log(f"[RELOAD] {err}")
        traceback.print_exc()
        return err



# Now we can safely import ctypes and other C-extensions
import ctypes
import json
import pickle
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

import mmap
import struct

class SimulatorManager:
    def __init__(self, bridge_dir):
        self.bridge_dir = bridge_dir
        self.shm_path = os.path.join(self.bridge_dir, "tmp/sts2_sim_shm")
        self.shm_size = 10 * 1024 * 1024 # 10MB
        self.shm = None
        # Must match Rust encoder: GLOBAL_SIZE + COMBAT_SIZE + BOW_SIZE * 4 + 2
        # 512 + 384 + 611 * 4 + 2 = 3342
        self.tensor_size = 512 + 384 + 611 * 4 + 2
        
    def init_simulator(self, sim):
        # Sync vocab
        sim.set_vocabulary(CARD_VOCAB, MONSTER_VOCAB, POWER_VOCAB, BOSS_VOCAB, POTION_VOCAB)
        # Init SHM
        os.makedirs(os.path.dirname(self.shm_path), exist_ok=True)
        sim.init_shm(self.shm_path, self.shm_size)
        # Map in Python
        if os.path.exists(self.shm_path):
            f = open(self.shm_path, "r+b")
            self.shm = mmap.mmap(f.fileno(), self.shm_size)
            log(f"[SimulatorManager] Shared memory mapped at {self.shm_path}")

    def read_results(self):
        if not self.shm or np is None:
            return []
        
        try:
            self.shm.seek(0)
            num_results_data = self.shm.read(4)
            if not num_results_data or len(num_results_data) < 4:
                return []
            num_results = struct.unpack("<i", num_results_data)[0]
            
            results = []
            for _ in range(num_results):
                header = self.shm.read(4)
                if not header: break
                num_actions = struct.unpack("<i", header)[0]
                actions_data = self.shm.read(4 * num_actions)
                actions = list(struct.unpack(f"<{num_actions}i", actions_data))
                
                num_outcomes_data = self.shm.read(4)
                if not num_outcomes_data: break
                num_outcomes = struct.unpack("<i", num_outcomes_data)[0]
                
                outcomes = []
                for __ in range(num_outcomes):
                    prob_data = self.shm.read(4)
                    if not prob_data: break
                    prob = struct.unpack("<f", prob_data)[0]
                    tensor_data = self.shm.read(4 * self.tensor_size)
                    tensor = np.frombuffer(tensor_data, dtype=np.float32).copy()
                    outcomes.append((prob, tensor))
                
                results.append({
                    "actions": actions,
                    "outcomes": outcomes
                })
            return results
        except Exception as e:
            log(f"[SimulatorManager] Error reading SHM: {e}")
            return []

sim_manager = SimulatorManager(BRIDGE_DIR)

import datetime

# Configure JAX to not preallocate all memory and force CPU if needed
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["JAX_PLATFORMS"] = "cpu"

# Debug MLflow environment variables
log(f"[Python] Debug: RNAD_RUN_ID={os.environ.get('RNAD_RUN_ID')}")
log(f"[Python] Debug: RNAD_CHECKPOINT={os.environ.get('RNAD_CHECKPOINT')}")
log(f"[Python] Debug: RNAD_SEED={os.environ.get('RNAD_SEED')}")
log(f"[Python] Debug: RNAD_ROUTE={os.environ.get('RNAD_ROUTE')}")
mask_card_skip = os.environ.get("RNAD_MASK_CARD_SKIP") == "true"
if mask_card_skip:
    log("Initialized mask_card_skip from environment: true")

offline_enabled = "--offline" in sys.argv or os.environ.get("RNAD_OFFLINE") == "true"
if offline_enabled:
    log("Offline training mode enabled.")

# Placeholders for deferred imports
jax = None
jnp = None
np = None
RNaDLearner = None
RNaDConfig = None
ExperimentManager = None
load_pretrained_embeddings = None
initialization_event = threading.Event()
initialization_lock = threading.Lock()

def do_deferred_imports():
    global jax, jnp, np, RNaDLearner, RNaDConfig, ExperimentManager, load_pretrained_embeddings
    if jax is None:
        import numpy as np_mod
        import jax as jax_mod
        import jax.numpy as jnp_mod
        from src.rnad import RNaDLearner as Learner, RNaDConfig as Config, load_pretrained_embeddings
        from experiment import ExperimentManager as ExpManager
        
        jax = jax_mod
        jnp = jnp_mod
        np = np_mod
        RNaDLearner = Learner
        RNaDConfig = Config
        ExperimentManager = ExpManager
        # Also need to assign the imported function to the global variable
        # Since it was imported as its original name in the from src.rnad import line,
        # and we didn't alias it, it's already in the local scope, but we need
        # it in the global scope for other functions.
        # Actually, let's just make sure it's in the global scope.
        globals()['load_pretrained_embeddings'] = load_pretrained_embeddings
        print("[Python] Deferred imports completed.")
        
        # Log JAX devices to confirm GPU usage
        try:
            import jax
            print(f"[Python] JAX devices: {jax.devices()}")
            print(f"[Python] JAX default backend: {jax.default_backend()}")
        except Exception as e:
            print(f"[Python] Error checking JAX devices: {e}")

# Global state
log(f"--- RNAD_BRIDGE MODULE IMPORTED --- PID: {os.getpid()}")
learning_active = False
can_continue_status = None

# Trajectory and Training Worker globals (need to be defined before preservation logic)
experience_queue = queue.Queue()
current_trajectory = []
training_worker = None
deferred_chunk = None
history = []

# Potion Vocabulary Mapping
POTION_VOCAB = {
    "UNKNOWN": 0,
    "empty": 0,
    "Fire Potion": 1,
    "Explosive Potion": 2,
    "FearPotion": 3,
    "Strength Potion": 4,
    "Dexterity Potion": 5,
    "Block Potion": 6,
    "Speed Potion": 7,
    "LiquidBronze": 8,
    "BottledCloud": 9,
    "Regen Potion": 10,
    "Swift Potion": 11,
    "Poison Potion": 12,
    "Weak Potion": 13,
    "ColorlessPotion": 14,
    "CultistPotion": 15,
    "FruitJuice": 16,
    "BloodPotion": 17,
    "ElixirPotion": 18,
    "HeartOfIron": 19,
    "GhostInAJar": 20,
    "Ambrosia": 21,
    "BlessingOfTheForge": 22,
    "DuplicationPotion": 23,
    "EssenceOfSteel": 24,
    "LiquidMemories": 25,
    "PotionOfCapacity": 26,
}

# --- Save Data Backup & Restoration Logic ---
SOURCE_A = "/home/ubuntu/.local/share/SlayTheSpire2/steam/76561198725031675/modded/profile1/saves"
SOURCE_B = "/home/ubuntu/.local/share/Steam/userdata/764765947/2868840/remote/modded/profile1/saves"
BACKUP_ROOT = os.path.expanduser("~/sts2_backups")

class BackupManager:
    def __init__(self, backup_root):
        self.backup_root = backup_root
        self.stack = [] # List of {path: str, retry_count: int, reward: float, trials: list}
        self.max_retries = 3
        self.hp_loss_history = [] # Track HP loss for each trial in the current encounter
        self.current_trial_actions = [] # Actions taken since the last backup or restore
        if not os.path.exists(self.backup_root):
            os.makedirs(self.backup_root, exist_ok=True)
            
    def clear(self):
        """Clear all backups in the stack."""
        log(f"[BackupManager] Clearing stack of size {len(self.stack)}")
        self.stack = []
        self.hp_loss_history = []
        self.current_trial_actions = []

    def _are_saves_identical(self, backup_dir):
        """Compare current saves with a backup directory."""
        appdata_backup = os.path.join(backup_dir, "AppData")
        userdata_backup = os.path.join(backup_dir, "UserData")
        
        # Check AppData
        if os.path.exists(SOURCE_A) and os.path.exists(appdata_backup):
            # Check only files, not metadata/directories
            dcomp = filecmp.dircmp(SOURCE_A, appdata_backup)
            if dcomp.left_only or dcomp.right_only or dcomp.diff_files:
                return False
        elif os.path.exists(SOURCE_A) != os.path.exists(appdata_backup):
            return False
            
        # Check UserData
        if os.path.exists(SOURCE_B) and os.path.exists(userdata_backup):
            dcomp = filecmp.dircmp(SOURCE_B, userdata_backup)
            if dcomp.left_only or dcomp.right_only or dcomp.diff_files:
                return False
        elif os.path.exists(SOURCE_B) != os.path.exists(userdata_backup):
            return False
            
        return True

    def backup(self, current_reward):
        """Create a backup if the state has changed."""
        if self.stack:
            last_backup = self.stack[-1]["path"]
            if (self._are_saves_identical(last_backup)):
                log(f"[BackupManager] Saves identical to latest backup. Checking combat status.")
                # Return True only if the last one was already a combat save.
                return self._is_combat_save(last_backup)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.backup_root, f"backup_{timestamp}")
        
        try:
            os.makedirs(os.path.join(backup_dir, "AppData"), exist_ok=True)
            os.makedirs(os.path.join(backup_dir, "UserData"), exist_ok=True)
            
            if os.path.exists(SOURCE_A):
                shutil.copytree(SOURCE_A, os.path.join(backup_dir, "AppData"), dirs_exist_ok=True)
            if os.path.exists(SOURCE_B):
                shutil.copytree(SOURCE_B, os.path.join(backup_dir, "UserData"), dirs_exist_ok=True)
                
            metadata = {
                "retry_count": 0,
                "accumulated_reward": float(current_reward)
            }
            with open(os.path.join(backup_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)
                
            self.stack.append({
                "path": backup_dir,
                "retry_count": 0,
                "reward": current_reward,
                "trials": []
            })
            log(f"[BackupManager] Created new backup at {backup_dir}. Stack size: {len(self.stack)}")
            # New backup point: clear path history so we diversify from this point onwards
            self.current_trial_actions = []
            # Verify if the new backup is a combat save
            is_combat = self._is_combat_save(backup_dir)
            if not is_combat:
                log(f"[BackupManager] Warning: Created backup is not a combat save. Will retry next turn.")
            return is_combat
        except Exception as e:
            log(f"[BackupManager] ERROR during backup: {e}")
            return False

    def _is_combat_save(self, backup_dir):
        """Check if the backup contains an active combat state."""
        # The saves are directly under AppData or UserData in our backup structure
        save_path = os.path.join(backup_dir, "AppData", "current_run.save")
        if not os.path.exists(save_path):
            save_path = os.path.join(backup_dir, "UserData", "current_run.save")
            if not os.path.exists(save_path):
                return False
        
        try:
            with open(save_path, "r") as f:
                data = json.load(f)
                pfr = data.get("pre_finished_room")
                if pfr is None:
                    # In StS2, pre_finished_room: null means middle of a room (active combat)
                    return True
                
                # is_pre_finished: true means rewards are showing or battle is over.
                # We want active battle (false).
                return not pfr.get("is_pre_finished", True)
        except Exception as e:
            log(f"[BackupManager] _is_combat_save: Error checking {save_path}: {e}")
            return False

    def record_hp_loss(self, hp_loss):
        self.hp_loss_history.append(hp_loss)
        log(f"[BackupManager] Recorded Trial HP Loss: {hp_loss}. History: {self.hp_loss_history}")

    def check_hp_performance(self, hp_loss):
        """Check if the current HP loss is in the top 50% of history."""
        if not self.hp_loss_history:
            return True, 0, self.max_retries
        
        # Sort and find median
        sorted_history = sorted(self.hp_loss_history)
        n = len(sorted_history)
        
        # Median: lower half is better
        median_idx = (n - 1) // 2
        median_val = sorted_history[median_idx]
        
        is_top_50 = hp_loss <= median_val
        
        retry_count = 0
        if self.stack:
            retry_count = self.stack[-1]["retry_count"]
            
        log(f"[BackupManager] HP Performance Check: loss={hp_loss}, median={median_val}, is_top_50={is_top_50}, retries={retry_count}/{self.max_retries}")
        return is_top_50, retry_count, self.max_retries

    def record_action(self, action_idx):
        """Record an action in the current trial."""
        self.current_trial_actions.append(int(action_idx))

    def save_trial_and_reset(self):
        """Save the current sequence of actions as a completed trial for the top backup."""
        if self.stack and self.current_trial_actions:
            latest = self.stack[-1]
            latest["trials"].append(list(self.current_trial_actions))
            log(f"[BackupManager] Saved trial of length {len(self.current_trial_actions)} to backup {os.path.basename(latest['path'])}. Total trials: {len(latest['trials'])}")
        self.current_trial_actions = []

    def get_penalized_actions(self):
        """Return a set of actions that have been taken in past trials for the current situation."""
        if not self.stack:
            return set()
        
        latest = self.stack[-1]
        trials = latest.get("trials", [])
        if not trials:
            return set()
        
        current_len = len(self.current_trial_actions)
        penalized = set()
        for trial in trials:
            # Check if this trial matches our current path so far
            if len(trial) > current_len and trial[:current_len] == self.current_trial_actions:
                penalized.add(trial[current_len])
        
        return penalized

    def restore(self, force=False):
        """Restore the latest valid backup, backtracking if necessary."""
        while self.stack:
            latest = self.stack[-1]
            
            # Check if this backup is a valid combat save.
            # If not, it means we took a backup at a reward screen or map by mistake.
            if not self._is_combat_save(latest["path"]):
                log(f"[BackupManager] Skipping non-combat/finished backup: {os.path.basename(latest['path'])}. Backtracking...")
                self.stack.pop()
                if os.path.exists(latest["path"]):
                    try:
                        shutil.rmtree(latest["path"])
                    except Exception as e:
                        log(f"[BackupManager] Warning: failed to delete skipped backup: {e}")
                continue

            if latest["retry_count"] < self.max_retries or force:
                # Perform restoration
                retry_msg = f"Retry {latest['retry_count']+1}/{self.max_retries}" if not force else "FORCED Retry"
                log(f"[BackupManager] Restoring {os.path.basename(latest['path'])} ({retry_msg})")
                
                try:
                    appdata_backup = os.path.join(latest["path"], "AppData")
                    userdata_backup = os.path.join(latest["path"], "UserData")
                    
                    if os.path.exists(appdata_backup):
                        shutil.copytree(appdata_backup, SOURCE_A, dirs_exist_ok=True)
                    if os.path.exists(userdata_backup):
                        shutil.copytree(userdata_backup, SOURCE_B, dirs_exist_ok=True)
                    
                    latest["retry_count"] += 1
                    return latest["reward"]
                except Exception as e:
                    log(f"[BackupManager] ERROR during restore: {e}")
                    return None
            else:
                log(f"[BackupManager] Backup {os.path.basename(latest['path'])} exhausted (tried {self.max_retries} times). Backtracking...")
                self.stack.pop()
        
        log(f"[BackupManager] FAILED: All backups exhausted or stack empty (stack size: {len(self.stack)}). Proceeding.")
        return None

# Initialize BackupManager
backup_manager = BackupManager(BACKUP_ROOT)
is_restoring = False

def trigger_backup():
    """Top-level function called from Rust bridge."""
    global reward_tracker
    return backup_manager.backup(reward_tracker.session_cumulative_reward)

def record_hp_loss(val):
    """Top-level function called from Rust bridge."""
    backup_manager.record_hp_loss(int(val))

def check_hp_performance(val):
    """Top-level function called from Rust bridge. Returns JSON string."""
    is_top_50, retry_count, max_retries = backup_manager.check_hp_performance(int(val))
    return json.dumps({
        "is_top_50": is_top_50,
        "retry_count": retry_count,
        "max_retries": max_retries
    })

def trigger_restore(force=False):
    """Top-level function called from Rust bridge or internally."""
    global is_restoring, deferred_chunk
    global reward_tracker, experience_queue, current_trajectory, history
    
    # Check for deferred_chunk (steps reached unroll_length but waiting for next_step)
    if deferred_chunk:
        log(f"[Python] /restore: Finalizing deferred_chunk (len {len(deferred_chunk['steps'])}) without next_step before restoration.")
        # Marking the last step as terminal for bootstrapping purposes (as requested by user)
        deferred_chunk['steps'][-1]['done'] = 1.0
        experience_queue.put({"steps": list(deferred_chunk['steps']), "next_step": None})
        deferred_chunk = None

    # Flush current trajectory before restoration
    if current_trajectory:
        log(f"[Python] /restore: Flushing current_trajectory of length {len(current_trajectory)} before restoration.")
        # Marking the last step as terminal for bootstrapping purposes (as requested by user)
        current_trajectory[-1]['done'] = 1.0
        experience_queue.put(list(current_trajectory))
        current_trajectory = []
    
    # NEW: Flush raw trajectory logger as well
    if raw_logger:
        raw_logger.flush(force_terminal=True)

    # NEW: Save current trial actions for diversification before restoration
    backup_manager.save_trial_and_reset()

    restored_reward = backup_manager.restore(force=force)
    if restored_reward is not None:
        is_restoring = True # Set flag so predict_action knows to wait for MainMenu
        if raw_logger:
            raw_logger.reset_ui = True
        reward_tracker.session_cumulative_reward = restored_reward
        # Mark combat not initialized to allow re-initialization in the next step
        reward_tracker.combat_initialized = False
        log(f"[Python] /restore: Restored reward tracker to {restored_reward:.2f}. is_restoring set to True.")
        return True
    return False

# Bridge state trackers
# Reward and Game State Tracking
class RewardTracker:
    def __init__(self):
        self.last_processed_floor: int = -1
        self.last_player_hp: int = 0
        self.last_total_enemy_hp: int = 0
        self.last_reward_floor: int = -1
        self.last_selected_reward_idx: int | None = None
        self.skipped_reward_indices: set[int] = set()
        self.session_cumulative_reward: float = 0.0
        self.episode_end_recorded: bool = False
        self.combat_initialized: bool = False
        self.last_state_type: str | None = None
        self.last_upgraded_count: int = -1
        self.last_total_cards: int = -1
        self.last_potion_count: int = -1
        self.was_elite: bool = False
        self.was_boss: bool = False

    def reset_for_new_run(self):
        """Reset state when returning to main menu or starting a fresh run."""
        self.last_processed_floor = -1
        self.last_player_hp = 0
        self.last_total_enemy_hp = 0
        self.last_reward_floor = -1
        self.last_selected_reward_idx = None
        self.skipped_reward_indices = set()
        self.session_cumulative_reward = 0.0
        self.episode_end_recorded = False
        self.combat_initialized = False
        self.last_state_type = None
        self.last_upgraded_count = -1
        self.last_total_cards = -1
        self.last_potion_count = -1
        self.was_elite = False
        self.was_boss = False
        backup_manager.clear()
        log("RewardTracker: Full reset for new run.")

    def reset_for_next_episode(self):
        """Reset per-episode flags but maybe keep some session info if needed."""
        self.episode_end_recorded = False
        self.session_cumulative_reward = 0.0
        self.skipped_reward_indices = set()
        self.combat_initialized = False
        # Do NOT reset last_processed_floor here as it might be used across screens
        log("RewardTracker: Per-episode reset.")

    def initialize_combat(self, hp, enemy_hp):
        """Initialize combat trackers to avoid the start-of-combat penalty."""
        self.last_player_hp = hp
        self.last_total_enemy_hp = enemy_hp
        self.combat_initialized = True
        log(f"RewardTracker: Combat initialized (Player HP: {hp}, Enemy HP: {enemy_hp})")

# Preserve RewardTracker state across re-imports
if 'rnad_bridge' in sys.modules:
    old_mod = sys.modules['rnad_bridge']
    if hasattr(old_mod, 'reward_tracker'):
        old_tracker = old_mod.reward_tracker
        reward_tracker = RewardTracker()
        reward_tracker.last_processed_floor = getattr(old_tracker, 'last_processed_floor', -1)
        reward_tracker.last_player_hp = getattr(old_tracker, 'last_player_hp', 0)
        reward_tracker.last_total_enemy_hp = getattr(old_tracker, 'last_total_enemy_hp', 0)
        reward_tracker.last_reward_floor = getattr(old_tracker, 'last_reward_floor', -1)
        reward_tracker.last_selected_reward_idx = getattr(old_tracker, 'last_selected_reward_idx', None)
        reward_tracker.skipped_reward_indices = getattr(old_tracker, 'skipped_reward_indices', set())
        reward_tracker.session_cumulative_reward = getattr(old_tracker, 'session_cumulative_reward', 0.0)
        reward_tracker.episode_end_recorded = getattr(old_tracker, 'episode_end_recorded', False)
        reward_tracker.combat_initialized = getattr(old_tracker, 'combat_initialized', False)
        log("Preserved RewardTracker state.")
    else:
        reward_tracker = RewardTracker()
else:
    reward_tracker = RewardTracker()

# Combat Prediction Verification
# Preserve globals across re-imports if already in sys.modules
if 'rnad_bridge' in sys.modules:
    old_mod = sys.modules['rnad_bridge']
    command_queue = getattr(old_mod, 'command_queue', queue.Queue())
    initialized = getattr(old_mod, 'initialized', False)
    # learner = getattr(old_mod, 'learner', None)
    learner = None # Force re-init with new code
    config = getattr(old_mod, 'config', None)
    np = getattr(old_mod, 'np', None)
    
    # NEW: Preserve training-related state
    experience_queue = getattr(old_mod, 'experience_queue', experience_queue)
    current_trajectory = getattr(old_mod, 'current_trajectory', current_trajectory)
    training_worker = getattr(old_mod, 'training_worker', None)
    deferred_chunk = getattr(old_mod, 'deferred_chunk', None)
    
    # NEW: Preserve predict_action state
    history = getattr(old_mod, 'history', [])
    
    log(f"Preserved state from existing rnad_bridge module. Queue size: {experience_queue.qsize() if hasattr(experience_queue, 'qsize') else 'unknown'}")
else:
    log("rnad_bridge module fresh load.")
    command_queue = queue.Queue()
    initialized = False
    learner = None
    config = None
    np = None

last_activity_time = time.time()
_predict_step = None

current_seed = os.environ.get("RNAD_SEED")
if current_seed:
    log(f"Initialized current_seed from environment: {current_seed}")

route_mode = os.environ.get("RNAD_ROUTE") == "true"
if route_mode:
    log("Initialized route_mode from environment: true")

# Card Vocabulary Mapping
CARD_VOCAB = {
    "UNKNOWN": 0,
    "STRIKE_IRONCLAD": 1,
    "STRIKE": 1, # Alias for base Strike
    "DEFEND_IRONCLAD": 2,
    "DEFEND": 2, # Alias for base Defend
    "BASH": 3,
    "ANGER": 4,
    "BODY_SLAM": 5,
    "EXPECT_A_FIGHT": 6,
    "EXPECT_AFIGHT": 6, # Aliased for safety
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
    "THUNDERCLAP": 16,
    "TRUE_GRIT": 17,
    "TWIN_STRIKE": 18,
    "WARCRY": 19,
    "WILD_STRIKE": 20,
    "ARMAMENTS": 21,
    "BLOOD_FOR_BLOOD": 22,
    "BLOOD_LETTING": 23,
    "BLOODLETTING": 23, # Alias for game ID
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
    "ASCENDERS_BANE": 77,
    "BREAKTHROUGH": 78
}

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
    "FLAME_BARRIER": 19,
    "MINION": 20
}

BOSS_VOCAB = {
    "UNKNOWN": 0,
    "THE_GHOST": 12
}

MONSTER_VOCAB = {
    "UNKNOWN": 0,
    "SLIME_M": 1, "SLIME_L": 2, "SLIME_S": 3,
    "GREMLIN_FAT": 4, "GREMLIN_TSAR": 5, "GREMLIN_WIZ": 6, "GREMLIN_SHIELD": 7, "GREMLIN_SNEAK": 8,
    "CULTIST": 9, "JAW_WORM": 10, "LOUSE_RED": 11, "LOUSE_GREEN": 12,
    "SLAVER_BLUE": 13, "SLAVER_RED": 14, "SCRATCHER": 15,
    "SENTRY": 16, "NOB": 17, "LAGAVULIN": 18,
    "BOOK_OF_STABBING": 19, "SLAYER_STATUE": 20, "NEMESIS": 21,
    "AUTOMATON_MINION": 22, "TORCH_HEAD": 23, "ORB_CORE": 24
}

# --- Vocabulary Expansion Logic ---
def expand_vocab(base_vocab, new_ids):
    current_ids = set(base_vocab.keys())
    # Find the next available index
    next_idx = int(max(base_vocab.values()) + 1 if base_vocab else 0)
    for id_str in sorted(new_ids):
        if id_str not in current_ids:
            base_vocab[id_str] = next_idx
            next_idx += 1
    return base_vocab, next_idx

# Load extracted IDs if available
GAME_IDS_PATH = "/home/ubuntu/src/R-NaD-StS2/R-NaD/scripts/game_ids.json"
if os.path.exists(GAME_IDS_PATH):
    try:
        with open(GAME_IDS_PATH, 'r') as f:
            game_ids_data = json.load(f)
        
        CARD_VOCAB, _ = expand_vocab(CARD_VOCAB, game_ids_data.get("CARDS", []))
        RELIC_VOCAB, _ = expand_vocab(RELIC_VOCAB, game_ids_data.get("RELICS", []))
        POWER_VOCAB, _ = expand_vocab(POWER_VOCAB, game_ids_data.get("POWERS", []))
        MONSTER_VOCAB, _ = expand_vocab(MONSTER_VOCAB, game_ids_data.get("MONSTERS", []))
        # Bosses in StS2 are defined in Encounters
        BOSS_VOCAB, _ = expand_vocab(BOSS_VOCAB, game_ids_data.get("ENCOUNTERS", []))
        BOSS_VOCAB["Unknown"] = 0 # Safety fallback
        
        log(f"Expanded vocabularies from {GAME_IDS_PATH}")
        log(f"CARD_VOCAB: {len(CARD_VOCAB)}, MONSTER_VOCAB: {len(MONSTER_VOCAB)}")
    except Exception as e:
        log(f"Error expanding vocabularies: {e}")

# Re-calculate or fix vocab sizes for the model
VOCAB_SIZE = max(max(CARD_VOCAB.values()) + 1, 600)
RELIC_VOCAB_SIZE = max(max(RELIC_VOCAB.values()) + 1, 300)
POWER_VOCAB_SIZE = max(max(POWER_VOCAB.values()) + 1, 280)
BOSS_VOCAB_SIZE = max(max(BOSS_VOCAB.values()) + 1, 100)
MONSTER_VOCAB_SIZE = max(max(MONSTER_VOCAB.values()) + 1, 128)


def get_monster_idx(monster_id):
    if not monster_id:
        assert False, f"monster_id is missing or empty"
    if isinstance(monster_id, dict):
        monster_id = monster_id.get("id") or monster_id.get("name")
    if not monster_id:
        assert False, "monster_id is missing or empty after dict extraction"
    mid = str(monster_id).upper().replace(" ", "_")
    
    if mid in MONSTER_VOCAB:
        return MONSTER_VOCAB[mid]
        
    # Substring search
    for k in MONSTER_VOCAB.keys():
        if mid in k or k in mid:
            return MONSTER_VOCAB[k]
            
    vocab_sample = list(MONSTER_VOCAB.keys())[:20]
    assert mid in MONSTER_VOCAB, f"Unknown monster_id: {monster_id} (mapped to {mid}). Vocab size: {len(MONSTER_VOCAB)}. Sample keys: {vocab_sample}"
    return MONSTER_VOCAB[mid]

def get_boss_idx(boss_id):
    if not boss_id:
        assert False, f"boss_id is missing or empty"
    if isinstance(boss_id, dict):
        boss_id = boss_id.get("id") or boss_id.get("name")
    if not boss_id:
        assert False, "boss_id is missing or empty after dict extraction"
    bid = str(boss_id).upper().replace(" ", "_")
    assert bid in BOSS_VOCAB, f"Unknown boss_id: {boss_id} (mapped to {bid})"
    return BOSS_VOCAB[bid]

def get_card_idx(card_id):
    if not card_id:
        # Some calls might pass empty strings for empty slots, but we should handle that at the call site or treat it as an error as requested.
        assert False, f"card_id is missing or empty"
    if isinstance(card_id, dict):
        card_id = card_id.get("id") or card_id.get("name")
    if not card_id:
        assert False, "card_id is missing or empty after dict extraction"
    # Clean up (e.g., remove name suffixes if any) and normalize spaces to underscores
    cid = str(card_id).split('+')[0].strip().upper().replace(" ", "_").replace("-", "_")
    # Remove punctuation for matching with vocab
    for char in "!?.(),":
        cid = cid.replace(char, "")
    
    if cid in CARD_VOCAB:
        return CARD_VOCAB[cid]
        
    # Robust fallbacks for base cards if the exact generic key is missing
    if cid == "STRIKE":
        for k in ["STRIKE_IRONCLAD", "STRIKE_SILENT", "STRIKE_DEFECT", "STRIKE_REGENT", "STRIKE_NECROBINDER"]:
            if k in CARD_VOCAB:
                return CARD_VOCAB[k]
    elif cid == "DEFEND":
        for k in ["DEFEND_IRONCLAD", "DEFEND_SILENT", "DEFEND_DEFECT", "DEFEND_REGENT", "DEFEND_NECROBINDER"]:
            if k in CARD_VOCAB:
                return CARD_VOCAB[k]
    
    # Substring search as a last resort for fuzzy matches
    for k in CARD_VOCAB.keys():
        if cid in k:
            return CARD_VOCAB[k]

    # Provide better diagnostics on failure
    vocab_sample = list(CARD_VOCAB.keys())[:20]
    assert cid in CARD_VOCAB, f"Unknown card_id: {card_id} (mapped to {cid}). Vocab size: {len(CARD_VOCAB)}. Sample keys: {vocab_sample}"
    return CARD_VOCAB[cid]

def get_relic_idx(relic_id):
    if not relic_id:
        assert False, f"relic_id is missing or empty"
    if isinstance(relic_id, dict):
        relic_id = relic_id.get("id") or relic_id.get("name")
    if not relic_id:
        assert False, "relic_id is missing or empty after dict extraction"
    rid = str(relic_id).upper().replace(" ", "_")
    assert rid in RELIC_VOCAB, f"Unknown relic_id: {relic_id} (mapped to {rid})"
    return RELIC_VOCAB[rid]

def get_power_idx(power_id):
    if not power_id:
        assert False, f"power_id is missing or empty"
    if isinstance(power_id, dict):
        power_id = power_id.get("id") or power_id.get("name")
    if not power_id:
        assert False, "power_id is missing or empty after dict extraction"
    pid = str(power_id).upper().replace(" ", "_")
    assert pid in POWER_VOCAB, f"Unknown power_id: {power_id} (mapped to {pid})"
    return POWER_VOCAB[pid]

def encode_bow(card_list):
    do_deferred_imports()
    assert np is not None
    vec = np.zeros(VOCAB_SIZE, dtype=np.float32)
    if isinstance(card_list, str):
        log(f"WARNING: encode_bow received a string instead of a list: {card_list[:100]}...")
        if card_list.strip().startswith("["):
            try:
                import json
                card_list = json.loads(card_list)
            except Exception as e:
                log(f"ERROR: Failed to parse card_list string as JSON: {e}")
                return vec
        else:
            return vec
            
    if not card_list: return vec
    for cid in card_list:
        idx = get_card_idx(cid)
        if idx < VOCAB_SIZE:
            vec[idx] += 1.0
    return vec

# Config placeholder (will be initialized in load_model)
# config = None # Removed as it's handled by the preservation logic above
# learner = None # Removed as it's handled by the preservation logic above
rng_key = None

# VALID_TRAJECTORY_STATES defines states that should be recorded as experiences.
VALID_TRAJECTORY_STATES = {
    "combat", "map", "rewards", "event", "rest_site", "shop", 
    "treasure", "treasure_relics", "card_reward"
}

# Raw Trajectory Logging
TRAJECTORY_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD/trajectories"
if not os.path.exists(TRAJECTORY_DIR):
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)

class RawTrajectoryLogger:
    def __init__(self, trajectory_dir):
        self.trajectory_dir = trajectory_dir
        self.current_episode = []
        self.reset_ui = False
        self.step_id = 0
        self.lock = threading.Lock()

    def log_step(self, state_json, action_idx, probs, mask, reward, log_prob, predicted_v=0.0, logits=None, terminal=False, is_search=False):
        with self.lock:
            self.current_episode.append({
                "state_json": state_json,
                "action_idx": int(action_idx),
                "probs": probs.tolist() if hasattr(probs, "tolist") else list(probs),
                "logits": logits.tolist() if logits is not None and hasattr(logits, "tolist") else list(logits) if logits is not None else [],
                "mask": mask.tolist() if hasattr(mask, "tolist") else list(mask),
                "reward": float(reward),
                "log_prob": float(log_prob),
                "predicted_v": float(predicted_v),
                "terminal": terminal
            })
            self.step_id += 1
            if terminal:
                self.flush()

            # NEW: Write to live_state.json for real-time monitoring
            try:
                import json as json_lib
                live_data = {
                    "state": json_lib.loads(state_json) if isinstance(state_json, str) else state_json,
                    "action_idx": int(action_idx),
                    "probs": probs.tolist() if hasattr(probs, "tolist") else list(probs),
                    "predicted_v": float(predicted_v),
                    "reward": float(reward),
                    "cum_reward": float(reward_tracker.session_cumulative_reward),
                    "terminal": terminal,
                    "timestamp": time.time(),
                    "step_id": self.step_id,
                    "mask": mask.tolist() if hasattr(mask, "tolist") else list(mask),
                    "reset": self.reset_ui,
                    "is_search": bool(is_search)
                }
                self.reset_ui = False
                live_path = os.path.abspath(os.path.join(self.trajectory_dir, "../tmp/live_state.json"))
                os.makedirs(os.path.dirname(live_path), exist_ok=True)
                
                tmp_live_path = live_path + ".tmp"
                with open(tmp_live_path, "w") as f:
                    json_lib.dump(live_data, f)
                os.replace(tmp_live_path, live_path)
            except Exception:
                pass

    def flush(self, force_terminal=False):
        if not self.current_episode:
            return
        
        with self.lock:
            if force_terminal and self.current_episode:
                self.current_episode[-1]["terminal"] = True

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"traj_{timestamp}.json"
            filepath = os.path.join(self.trajectory_dir, filename)
            
            # Build semantic map for this episode
            semantic_map = get_semantic_map()
            
            try:
                with open(filepath, "w") as f:
                    json.dump({
                        "semantic_map": semantic_map,
                        "steps": self.current_episode
                    }, f)
                log(f"RawTrajectoryLogger: Saved episode segment with {len(self.current_episode)} steps to {filepath} (force_terminal={force_terminal})")
            except Exception as e:
                log(f"RawTrajectoryLogger: Error saving trajectory: {e}")
            
            self.current_episode = []

raw_logger = RawTrajectoryLogger(TRAJECTORY_DIR)

def get_semantic_map():
    return {
        "CARD_VOCAB": CARD_VOCAB,
        "RELIC_VOCAB": RELIC_VOCAB,
        "POWER_VOCAB": POWER_VOCAB,
        "BOSS_VOCAB": BOSS_VOCAB,
        "ACTION_SPACE": {
            "0-49": "Cards (up to 10 cards * 5 targets)",
            "50-74": "Potions (up to 5 potions * 5 targets)",
            "75": "End Turn",
            "76-85": "Select Reward",
            "86": "Proceed",
            "87": "Return to Main Menu",
            "88-89": "Room Selection",
            "90": "Confirm Selection / Skip (Grid)",
            "91": "Open Chest",
            "92-93": "Shop Interaction",
            "94-98": "Discard Potion",
            "99": "Wait"
        }
    }

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
    def __init__(self, learner, config, experiment_manager=None, step_count=0):
        super().__init__(daemon=True)
        self.learner = learner
        self.config = config
        self.experiment_manager = experiment_manager
        self.batch_buffer = []
        self.running = True
        self.step_count = step_count
        self.episode_game_overed_floors = []
        self.episode_game_overed_rewards = []
        self.last_known_mean_game_overed_floor: float | None = None
        self.last_known_mean_game_overed_reward: float | None = None
        self.lock = threading.Lock()
        self.buffer_lock = threading.Lock()
        self.last_error: str | None = None
        self.is_updating = False
        self.update_progress = 0
        self.update_total = 0

    def run(self):
        print("[Python] TrainingWorker started.")
        while self.running:
            try:
                # Wait for a trajectory segment
                trajectory = experience_queue.get(timeout=1.0)
                
                batch = None
                with self.buffer_lock:
                    self.batch_buffer.append(trajectory)
                    if len(self.batch_buffer) >= self.config.batch_size:
                        # Ensure constant batch size for JAX JIT stability
                        batch = self.batch_buffer[:self.config.batch_size]
                        self.batch_buffer = self.batch_buffer[self.config.batch_size:]
                
                if batch:
                    # Perform update outside the lock
                    self.perform_update(batch)
            except queue.Empty:
                continue
            except Exception as e:
                self.last_error = str(e)
                print(f"[Python] Error in TrainingWorker: {e}")
                traceback.print_exc()

    def record_game_over(self, floor, reward):
        with self.lock:
            self.episode_game_overed_floors.append(floor)
            self.episode_game_overed_rewards.append(reward)
            print(f"[Python] Recorded game over at floor {floor}, reward {reward:.2f}. Count: {len(self.episode_game_overed_floors)}")

    def perform_offline_training(self):
        with self.lock:
            if self.is_updating:
                log("[Python] Already updating, skipping offline training request.")
                return
            self.is_updating = True
        
        do_deferred_imports()
        log(f"[Python] Starting offline training from trajectories in {TRAJECTORY_DIR}...")
        
        try:
            import glob
            files = sorted(glob.glob(os.path.join(TRAJECTORY_DIR, "traj_*.json")))
            trajectories = []
            if files:
                for filepath in files:
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f)
                            steps = data.get("steps", [])
                            traj_segment = []
                            for idx, step in enumerate(steps):
                                state_json = step["state_json"]
                                state = json.loads(state_json)
                                state_dict = encode_state(state)
                                
                                # Recalculate reward
                                state_type = state.get("type", "unknown")
                                action_idx = step["action_idx"]
                                base_reward = compute_reward(state, state_type)
                                intermediate_reward = compute_intermediate_reward(state, state_type, action_idx)
                                reward = base_reward + intermediate_reward
                                
                                # Fallback for older trajectories missing log_prob
                                log_p = step.get("log_prob")
                                if log_p is None:
                                    # Calculate from probs if available
                                    if "probs" in step and action_idx < len(step["probs"]):
                                        log_p = float(np.log(max(step["probs"][action_idx], 1e-10)))
                                    else:
                                        log_p = 0.0 # Default fallback
                                        
                                traj_step = {
                                    "obs": state_dict,
                                    "act": action_idx,
                                    "rew": reward,
                                    "mask": np.array(step["mask"], dtype=np.float32),
                                    "log_prob": log_p,
                                    "probs_dist": np.array(step.get("probs", np.zeros(100)), dtype=np.float32),
                                    "predicted_v": float(step.get("predicted_v", 0.0)),
                                    "is_human": 0.0,
                                    "done": 1.0 if step.get("terminal") else 0.0
                                }
                                traj_segment.append(traj_step)
                                
                                if len(traj_segment) >= self.config.unroll_length:
                                    # Look ahead for bootstrapping step
                                    next_step = None
                                    if idx + 1 < len(steps):
                                        next_s_raw = steps[idx + 1]
                                        next_s_json = next_s_raw["state_json"]
                                        next_s_dict = encode_state(json.loads(next_s_json))
                                        next_step = {
                                            "obs": next_s_dict,
                                            "mask": np.array(next_s_raw["mask"], dtype=np.float32)
                                        }
                                    trajectories.append({"steps": list(traj_segment), "next_step": next_step})
                                    traj_segment = []
                            
                            if traj_segment:
                                trajectories.append({"steps": list(traj_segment), "next_step": None})
                    except Exception as e:
                        log(f"[Python] Error processing {filepath}: {e}")

            # Now parse human replays
            human_files = sorted(glob.glob("/mnt/nas/StS2/replay/human_play_*.jsonl"))
            human_trajectories = []
            if human_files:
                log(f"[Python] Starting offline training from {len(human_files)} human play files in /mnt/nas/StS2/replay...")
                for filepath in human_files:
                    try:
                        with open(filepath, "r") as f:
                            lines = f.readlines()
                            
                        # process human steps...
                        steps = []
                        for line in lines:
                            line = line.strip()
                            if not line: continue
                            step_data = json.loads(line)
                            state = step_data.get("state")
                            if not state: continue
                            action_id = step_data.get("action_id")
                            if action_id is None: continue
                            
                            try:
                                action_idx = int(action_id)
                            except ValueError:
                                continue # skip invalid actions
                                
                            state_type = state.get("type", "unknown")
                            reward = compute_reward(state, state_type) + compute_intermediate_reward(state, state_type, action_idx)
                            
                            terminal = (state_type == "game_over")
                            
                            steps.append({
                                "state": state,
                                "action_idx": action_idx,
                                "reward": reward,
                                "terminal": terminal
                            })
                            
                        # convert to trajectories
                        traj_segment = []
                        for idx, step in enumerate(steps):
                            state = step["state"]
                            state_dict = encode_state(state)
                            action_idx = step["action_idx"]
                            
                            mask = get_action_mask(state)
                            
                            # For human data, we create a uniform distribution over valid actions for probs_dist placeholder
                            probs_placeholder = np.zeros(100, dtype=np.float32)
                            valid_indices = np.where(mask > 0)[0]
                            if len(valid_indices) > 0:
                                probs_placeholder[valid_indices] = 1.0 / len(valid_indices)

                            traj_step = {
                                "obs": state_dict,
                                "act": action_idx,
                                "rew": step["reward"],
                                "mask": np.array(mask, dtype=np.float32),
                                "log_prob": 0.0, # Human data has fixed log prob 0
                                "probs_dist": probs_placeholder,
                                "predicted_v": 0.0,
                                "is_human": 1.0,
                                "done": 1.0 if step["terminal"] else 0.0
                            }
                            traj_segment.append(traj_step)
                            
                            if len(traj_segment) >= self.config.unroll_length or step["terminal"]:
                                next_step = None
                                if not step["terminal"] and idx + 1 < len(steps):
                                    next_s = steps[idx + 1]["state"]
                                    next_s_dict = encode_state(next_s)
                                    next_mask = get_action_mask(next_s)
                                    next_step = {
                                        "obs": next_s_dict,
                                        "mask": np.array(next_mask, dtype=np.float32)
                                    }
                                human_trajectories.append({"steps": list(traj_segment), "next_step": next_step})
                                traj_segment = []
                                
                        if traj_segment:
                            human_trajectories.append({"steps": list(traj_segment), "next_step": None})
                    except Exception as e:
                        log(f"[Python] Error processing human replay {filepath}: {e}")

            all_updates = []
            
            # Combine standard trajectories
            if trajectories:
                all_updates.extend(trajectories)
                
            # Multiple passes for human data
            if human_trajectories:
                log(f"[Python] Enhancing {len(human_trajectories)} human play segments with multiple passes.")
                for epoch in range(5):
                    all_updates.extend(human_trajectories)

            if not all_updates:
                log("[Python] No valid trajectory segments found for offline training.")
                return

            log(f"[Python] Found {len(all_updates)} total trajectory segments. Performing updates...")
            
            batch_size = self.config.batch_size
            total_batches = len(all_updates) // batch_size
            with self.lock:
                self.update_total = total_batches
                self.update_progress = 0
            for i in range(0, len(all_updates), batch_size):
                batch = all_updates[i : i + batch_size]
                if len(batch) < batch_size:
                    continue 
                
                self.perform_update(batch, increment_step=False, reset_updating=False)
                
                log(f"[Python] Offline update {i // batch_size + 1}/{(len(all_updates) // batch_size)} done.")
                with self.lock:
                    self.update_progress += 1

            log("[Python] Offline training complete.")

        finally:
            with self.lock:
                self.is_updating = False
                self.update_progress = 0
                self.update_total = 0

    def perform_update(self, batch, increment_step=True, reset_updating=True):
        with self.lock:
            self.is_updating = True
            if increment_step:
                self.update_progress = 0
                self.update_total = 1
        
        print("[Python] TrainingWorker: Bridge is performing an update. Waiting...")

        try:
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
                "state_type": [],
                "head_type": []
            }
            padded_act = []
            padded_rew = []
            padded_mask = []
            padded_probs_dist = []
            padded_pred_v = []
            padded_is_human = []
            padded_log_prob = []
            padded_done = []
            padded_next_obs_dict = {
                "global": [], "combat": [], "draw_bow": [], "discard_bow": [],
                "exhaust_bow": [], "master_bow": [], "map": [], "event": [], "state_type": [], "head_type": []
            }
            padded_next_mask = []
            valid_mask = []

            for traj_item in batch:
                # traj_item can be a list of steps (old) or dict with 'steps' and 'next_step' (new)
                if isinstance(traj_item, dict) and "steps" in traj_item:
                    traj = traj_item["steps"]
                    next_step = traj_item.get("next_step")
                elif isinstance(traj_item, list):
                    traj = traj_item
                    next_step = None
                else:
                    log(f"[Python] Warning: Unknown trajectory format: {type(traj_item)}")
                    continue
                
                l = len(traj)
                if l == 0: continue
                
                # Extract observations
                obs_traj = [t['obs'] for t in traj]
                
                # Pad each element in the dict
                for key in padded_obs_dict.keys():
                    # Default head_type to 5 (Reward/Selection) for old trajectories
                    default_val = np.int32(5) if key == "head_type" else (np.int32(2) if key == "state_type" else None)
                    
                    if default_val is not None:
                        val_traj = [o.get(key, default_val) for o in obs_traj]
                        val_traj += [default_val] * (max_len - l)
                    else:
                        val_traj = [o[key] for o in obs_traj]
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

                # Pad probs_dist
                probs_traj = [t.get('probs_dist', np.zeros(100)) for t in traj]
                probs_traj += [np.zeros_like(probs_traj[0])] * (max_len - l)
                padded_probs_dist.append(probs_traj)

                # Pad predicted_v
                pv_traj = [t.get('predicted_v', 0.0) for t in traj]
                pv_traj += [0.0] * (max_len - l)
                padded_pred_v.append(pv_traj)

                # Pad is_human
                ih_traj = [t.get('is_human', 0.0) for t in traj]
                ih_traj += [0.0] * (max_len - l)
                padded_is_human.append(ih_traj)

                # Pad log_prob
                lp_traj = [t['log_prob'] for t in traj]
                lp_traj += [0.0] * (max_len - l)
                padded_log_prob.append(lp_traj)

                # Pad done
                done_traj = [t.get('done', 0.0) for t in traj]
                done_traj += [0.0] * (max_len - l)
                padded_done.append(done_traj)

                # Valid mask
                v_mask = [1.0] * l + [0.0] * (max_len - l)
                valid_mask.append(v_mask)

                # Next step for bootstrapping
                if next_step:
                    no = next_step['obs']
                    for key in padded_next_obs_dict.keys():
                        # Use same defaults as above for bootstrapping
                        default_val = np.int32(5) if key == "head_type" else (np.int32(2) if key == "state_type" else None)
                        if default_val is not None:
                            padded_next_obs_dict[key].append(no.get(key, default_val))
                        else:
                            padded_next_obs_dict[key].append(no[key])
                    padded_next_mask.append(next_step['mask'])
                else:
                    # Episode end or no next step available
                    for key in padded_next_obs_dict.keys():
                        if key == "head_type":
                            padded_next_obs_dict[key].append(np.int32(5))
                        elif key == "state_type":
                            padded_next_obs_dict[key].append(np.int32(2))
                        else:
                            padded_next_obs_dict[key].append(np.zeros_like(padded_obs_dict[key][0][0]))
                    padded_next_mask.append(np.zeros_like(padded_mask[0][0]))

            print("[Python] TrainingWorker: Padding done.")

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
            done = np.array(padded_done)
            valid = np.array(valid_mask)
            
            # Next obs/mask for bootstrapping (shape B, ...)
            jax_next_obs = {
                k: jnp.array(np.array(v)) for k, v in padded_next_obs_dict.items()
            }
            next_mask = jnp.array(np.array(padded_next_mask))

            batch = {
                'obs': jax_obs,
                'act': jnp.array(act.transpose(1, 0)),
                'rew': jnp.array(rew.transpose(1, 0)),
                'mask': jnp.array(mask.transpose(1, 0, 2)),
                'probs_dist': jnp.array(np.array(padded_probs_dist).transpose(1, 0, 2)),
                'predicted_v': jnp.array(np.array(padded_pred_v).transpose(1, 0)),
                'is_human': jnp.array(np.array(padded_is_human).transpose(1, 0)),
                'log_prob': jnp.array(log_prob.transpose(1, 0)),
                'done': jnp.array(done.transpose(1, 0)),
                'valid': jnp.array(valid.transpose(1, 0)),
                'next_obs': jax_next_obs,
                'next_mask': next_mask
            }

            print(f"[Python] TrainingWorker: Batch built. Shape: T={batch['rew'].shape[0]}, B={batch['rew'].shape[1]}")

            t_start = time.time()
            metrics = self.learner.update(batch, self.step_count)
            t_end = time.time()

            print(f"[Python] TrainingWorker: Update done in {t_end - t_start:.2f}s.")

            # Add mean game overed floor/reward if we have data
            with self.lock:
                if self.episode_game_overed_floors:
                    # Calculate mean of episodes that ended since last update
                    self.last_known_mean_game_overed_floor = sum(self.episode_game_overed_floors) / len(self.episode_game_overed_floors)
                    self.last_known_mean_game_overed_reward = sum(self.episode_game_overed_rewards) / len(self.episode_game_overed_rewards)
                    self.episode_game_overed_floors = [] # Clear for next update
                    self.episode_game_overed_rewards = []
                
                # Report the last known means to keep metrics visible in MLflow
                if self.last_known_mean_game_overed_floor is not None:
                    metrics['mean_game_overed_floor'] = self.last_known_mean_game_overed_floor
                if self.last_known_mean_game_overed_reward is not None:
                    metrics['mean_game_overed_reward'] = self.last_known_mean_game_overed_reward

            if increment_step:
                self.step_count += 1
                
                log_msg = f"[Python] Training Step {self.step_count}: Loss={metrics['loss']:.4f}, Policy Loss={metrics['policy_loss']:.4f}, Entropy Alpha={metrics['alpha']:.4f}"
                if 'mean_game_overed_floor' in metrics:
                    log_msg += f", Mean Game Overed Floor={metrics['mean_game_overed_floor']:.2f}"
                if 'mean_game_overed_reward' in metrics:
                    log_msg += f", Mean Game Overed Reward={metrics['mean_game_overed_reward']:.2f}"
                print(log_msg)
                
                if self.experiment_manager:
                    self.experiment_manager.log_metrics(self.step_count, metrics)

                if self.step_count % self.config.save_interval == 0:
                    checkpoint_path = f"/home/ubuntu/src/R-NaD-StS2/R-NaD/checkpoints/checkpoint_{self.step_count}.pkl"
                    if self.experiment_manager:
                        # Use experiment-specific subdir if possible
                        checkpoint_path = os.path.join(self.experiment_manager.checkpoint_dir, f"checkpoint_{self.step_count}.pkl")
                    
                    self.learner.save_checkpoint(checkpoint_path, self.step_count)
                    print(f"[Python] TrainingWorker: Saved checkpoint to {checkpoint_path}")
                    
                    if self.experiment_manager:
                        self.experiment_manager.log_checkpoint_artifact(self.step_count, checkpoint_path)
            else:
                log_msg = f"[Python] Offline Training (Step {self.step_count}): Loss={metrics['loss']:.4f}, Policy Loss={metrics['policy_loss']:.4f}, Entropy Alpha={metrics['alpha']:.4f}"
                print(log_msg)
                if self.experiment_manager:
                    # Log to the current step (likely 0 for pre-training)
                    self.experiment_manager.log_metrics(self.step_count, metrics)
        finally:
            if reset_updating:
                with self.lock:
                    self.is_updating = False
                    if increment_step:
                        self.update_progress = 0
                        self.update_total = 0

# training_worker = None # Handled at top now

def load_model(checkpoint_path=None):
    global learner, rng_key, training_worker, config, initialization_lock, jax, jnp
    
    with initialization_lock:
        do_deferred_imports()
        if initialization_event.is_set():
            return
    
    num_actions = 100
    # Create config with dynamic vocab sizes (already expanded at module level)
    config = RNaDConfig(
        card_vocab_size=VOCAB_SIZE,
        monster_vocab_size=MONSTER_VOCAB_SIZE,
        relic_vocab_size=RELIC_VOCAB_SIZE,
        power_vocab_size=POWER_VOCAB_SIZE
    )
    
    # Updated dummy observation for structured dictionary input
    # Ensure dummy observation matches VOCAB_SIZE etc.
    dummy_obs = {
        "global": jnp.zeros((1, 1, 512)),
        "combat": jnp.zeros((1, 1, 384)),
        "relic_ids": jnp.zeros((1, 1, 30)),
        "draw_bow": jnp.zeros((1, 1, VOCAB_SIZE)),
        "discard_bow": jnp.zeros((1, 1, VOCAB_SIZE)),
        "exhaust_bow": jnp.zeros((1, 1, VOCAB_SIZE)),
        "master_bow": jnp.zeros((1, 1, VOCAB_SIZE)),
        "map": jnp.zeros((1, 1, 2048)),
        "event": jnp.zeros((1, 1, 128)),
        "state_type": jnp.zeros((1, 1), dtype=jnp.int32),
        "head_type": jnp.zeros((1, 1), dtype=jnp.int32)
    }
    dummy_mask = jnp.ones((1, 1, num_actions))
    
    # Pass 0 as state_dim (it's not used by Learner anymore, it uses config/obs_dict)
    assert RNaDLearner is not None
    assert jax is not None
    assert ExperimentManager is not None
    
    learner = RNaDLearner(0, num_actions, config)
    rng_key = jax.random.PRNGKey(0)

    # Define JIT-compiled prediction step to avoid re-compilation
    global _predict_step
    @jax.jit
    def _predict_step(params, rng, state, mask, is_training):
        logits, value = learner.network.apply(params, rng, state, mask, is_training=is_training)
        return logits, value

    # Pre-warm JAX compilation for all state types
    log("[Python] Pre-warming JAX compilation for all state types...")
    t_warm_start = time.time()
    try:
        if learner.params is None:
            log("[Python] Initializing JAX model params for pre-warm...")
            learner.init(rng_key)
            
            # Load pretrained embeddings
            EMB_PATH = os.path.join(BRIDGE_DIR, "..", "text_encoder", "embeddings.pkl")
            log(f"[Python] Loading pretrained embeddings from {EMB_PATH}")
            learner.params = load_pretrained_embeddings(
                learner.params, EMB_PATH, 
                CARD_VOCAB, MONSTER_VOCAB, POWER_VOCAB, RELIC_VOCAB
            )
            learner.fixed_params = learner.params
            
            log(f"[Python] Initialized params keys: {list(learner.params.keys())}")
        
        # Pre-warm for each switch branch (combat:0, map:1, event-like:2, grid:3, hand:4)
        for st_idx in [0, 1, 2, 3, 4]:
            # Pre-warm for all history lengths up to seq_len
            # Actually, with padding we only need to warm for the fixed seq_len.
            # But let's keep it robust and warm for T=config.seq_len.
            T_warm = config.seq_len if config else 8
            temp_obs = jax.tree_util.tree_map(lambda x: jnp.repeat(x, T_warm, axis=0), dummy_obs) # (T, 1, ...)
            temp_obs["state_type"] = jnp.zeros((T_warm, 1), dtype=jnp.int32) + st_idx # (T, 1)
            dummy_mask_jit = jnp.zeros((T_warm, 1, num_actions), dtype=jnp.float32)
            
            # Split key for each pre-warm call to ensure stability
            predict_key, rng_key = jax.random.split(rng_key)
            # Pre-warm for all head types (0-5)
            for ht_idx in range(6):
                temp_obs_ht = temp_obs.copy()
                temp_obs_ht["head_type"] = jnp.zeros((T_warm, 1), dtype=jnp.int32) + ht_idx
                # Pre-warm for both training and inference modes
                for is_train_val in [True, False]:
                    _ = _predict_step(learner.params, predict_key, temp_obs_ht, dummy_mask_jit, is_train_val)
            log(f"[Python] JAX pre-warm for type {st_idx} complete.")
            
        log(f"[Python] All JAX pre-warms complete in {time.time() - t_warm_start:.2f}s")
    except Exception as e:
        log(f"[Python] Warning: JAX pre-warm failed: {e}")
        traceback.print_exc()

    # Initialize ExperimentManager
    exp_manager = None
    run_id = os.environ.get("RNAD_RUN_ID")
    try:
        exp_manager = ExperimentManager(experiment_name="R-NaD-StS2", log_checkpoints=True, run_id=run_id)
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
                # Extract step number and find the max based on mtime to handle step resets
                def get_mtime(path):
                    try:
                        return os.path.getmtime(path)
                    except OSError:
                        return 0
                
                latest_checkpoint = max(checkpoints, key=get_mtime)
                checkpoint_path = latest_checkpoint
                print(f"[Python] Auto-detected latest checkpoint (by mtime): {checkpoint_path}")

    step = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            step = learner.load_checkpoint(checkpoint_path)
            log(f"[Python] Loaded JAX model from {checkpoint_path} at step {step}")
            if learner.params is None:
                log("[Python] ERROR: learner.params is STILL None after load_checkpoint!")
            else:
                log(f"[Python] learner.params initialized with {len(learner.params)} modules")
        except Exception as e:
            log(f"[Python] Error loading checkpoint {checkpoint_path}: {e}")
            learner.init(rng_key)
            log("[Python] Falling back to new JAX model initialization")
    else:
        learner.init(rng_key)
        log("[Python] Initialized new JAX model")
    
    if learner.params is None:
        log("[Python] CRITICAL: learner.params is None at end of load_model!")
    
    # Start training worker if learning behavior is expected
    if training_worker is None:
        training_worker = TrainingWorker(learner, config, experiment_manager=exp_manager, step_count=step)
        training_worker.start()
    
    initialization_event.set()

def encode_state(state):
    do_deferred_imports()
    assert np is not None
    """Encodes the game state into a structured dictionary of NumPy arrays."""
    state_type = state.get("type", "unknown")
    
    # State type mapping for Experts (Backbone)
    # 0: Combat
    # 1: Map
    # 2: Event-like (Rewards, Event, Shop, Rest, Treasure, etc.)
    # 3: Grid Selection
    # 4: Hand Selection
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
        "hand_selection": 4,
        "combat_waiting": 0
    }
    if state_type not in type_map:
        assert False, f"Unknown state_type for type_map: {state_type}"
    st_idx = type_map[state_type]

    # Head type mapping (Actor/Critic heads)
    # 0: Combat
    # 1: Map
    # 2: Shop
    # 3: Rest Site
    # 4: Event
    # 5: Reward (Post-Combat Reward screens + selection screens)
    head_map = {
        "combat": 0,
        "combat_waiting": 0,
        "map": 1,
        "shop": 2,
        "rest_site": 3,
        "event": 4,
        "rewards": 5,
        "card_reward": 5,
        "treasure": 5,
        "treasure_relics": 5,
        "grid_selection": 5,
        "hand_selection": 5,
        "game_over": 5
    }
    if state_type not in head_map:
        assert False, f"Unknown state_type for head_map: {state_type}"
    head_idx = head_map[state_type]
    
    # Global features (Size 512)
    global_vec = np.zeros(512, dtype=np.float32)
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
    boss_id = state.get("boss") or "UNKNOWN"
    global_vec[20] = get_boss_idx(boss_id) / float(BOSS_VOCAB_SIZE)
    
    # Relics (up to 30)
    relics = state.get("relics", [])
    if not relics and player: # Fallback for combat state where player is a child
        relics = player.get("relics", [])
    
    relic_ids = np.zeros(30, dtype=np.float32)
    for i, rid in enumerate(relics[:30]):
        idx = get_relic_idx(rid) # get_relic_idx handles dicts and strings consistently
        relic_ids[i] = idx
        # Also keep multi-hot in global_vec for legacy/compatibility
        if 0 < idx and 30 + idx < 512:
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
        hand = state.get("hand") or []
        for i in range(min(len(hand), 10)):
            card = hand[i]
            base_idx = 10 + i * 10
            # Card ID index for embedding
            combat_vec[base_idx] = get_card_idx(card.get("id", ""))
            combat_vec[base_idx + 1] = 1.0 if card.get("isPlayable") else 0.0
            
            target_type = card.get("targetType", "None")
            # Map C# TargetType.ToString() to indices. Note: C# uses AnyEnemy and AllEnemies.
            tt_map = {
                "AnyEnemy": 1, "SingleEnemy": 1, 
                "AllEnemies": 2, "AllEnemy": 2, 
                "RandomEnemy": 3, 
                "None": 0, 
                "Self": 4
            }
            assert target_type in tt_map, f"Unknown targetType: {target_type}"
            combat_vec[base_idx + 2] = tt_map[target_type] / 10.0
            combat_vec[base_idx + 3] = card.get("cost", 0) / 5.0
            combat_vec[base_idx + 4] = card.get("baseDamage", 0) / 20.0
            combat_vec[base_idx + 5] = card.get("baseBlock", 0) / 20.0
            combat_vec[base_idx + 6] = card.get("magicNumber", 0) / 10.0
            combat_vec[base_idx + 7] = 1.0 if card.get("upgraded") else 0.0
            combat_vec[base_idx + 8] = card.get("currentDamage", 0) / 50.0
            combat_vec[base_idx + 9] = card.get("currentBlock", 0) / 50.0

        # Enemies (up to 5 enemies, 16 features each)
        # Offset to 110: 110 + 5*16 = 190 (fits before powers at 200)
        enemies = state.get("enemies", [])
        for i in range(min(len(enemies), 5)):
            enemy = enemies[i]
            base_idx = 110 + i * 16
            
            combat_vec[base_idx] = 1.0 # Alive flag restored
            combat_vec[base_idx + 1] = get_monster_idx(enemy.get("id")) # Enemy ID shifted
            combat_vec[base_idx + 2] = 1.0 if enemy.get("isMinion") else 0.0 # Minion flag shifted
            combat_vec[base_idx + 3] = enemy.get("hp", 0) / 200.0
            combat_vec[base_idx + 4] = enemy.get("maxHp", 1) / 200.0
            combat_vec[base_idx + 5] = enemy.get("block", 0) / 50.0
            
            intents = enemy.get("intents", [])
            for j in range(min(len(intents), 2)):
                intent = intents[j]
                intent_idx = base_idx + 6 + j * 4
                it_map = {
                    "Attack": 1, 
                    "Defense": 2, 
                    "Defend": 2, 
                    "AttackDefense": 3, 
                    "Buff": 4, 
                    "Debuff": 5, 
                    "StrongDebuff": 6, 
                    "DebuffStrong": 6, 
                    "Stun": 7, 
                    "StatusCard": 8, 
                    "Summon": 9,
                    "CardDebuff": 10,
                    "Heal": 11,
                    "Escape": 12,
                    "Hidden": 13,
                    "Sleep": 14,
                    "DeathBlow": 15,
                    "Unknown": 0
                }
                it_type = intent.get("type", "Unknown")
                if it_type not in it_map:
                    log(f"[Python] Warning: intent type: {it_type} not in it_map, falling back to Unknown")
                    it_type = "Unknown"
                combat_vec[intent_idx] = it_map[it_type] / 10.0
                combat_vec[intent_idx + 1] = intent.get("damage", 0) / 50.0
                combat_vec[intent_idx + 2] = intent.get("repeats", 1) / 5.0
                combat_vec[intent_idx + 3] = intent.get("count", 0) / 10.0
                log(f"[Python] Debug Enemy {i}: ID={enemy.get('id')}, Alive={combat_vec[base_idx]}, Minion={combat_vec[base_idx+2]}, Intent={it_type}")
                
        # Powers (starting at 200)
        # Player powers (up to 10, index 200-219)
        p_powers = player.get("powers", [])
        for i in range(min(len(p_powers), 10)):
            p = p_powers[i]
            base_idx = 200 + i * 2
            combat_vec[base_idx] = get_power_idx(p.get("id"))
            combat_vec[base_idx + 1] = p.get("amount", 0) / 10.0
            
        # Enemy powers (up to 5 enemies, 10 powers each, starting at 220)
        for i in range(min(len(enemies), 5)):
            e_powers = enemies[i].get("powers", [])
            enemy_base = 220 + i * 20
            for j in range(min(len(e_powers), 10)):
                p = e_powers[j]
                idx = enemy_base + j * 2
                combat_vec[idx] = get_power_idx(p.get("id"))
                combat_vec[idx + 1] = p.get("amount", 0) / 10.0
                
        # --- Predicted Features (starting at 320) ---
        combat_vec[320] = state.get("predicted_total_damage", 0) / 50.0
        combat_vec[321] = state.get("predicted_end_block", 0) / 50.0
        combat_vec[322] = 1.0 if state.get("surplus_block") else 0.0
        
        log(f"[Python] Predicted Damage: {state.get('predicted_total_damage', 0)}, Predicted Block: {state.get('predicted_end_block', 0)}, Surplus: {state.get('surplus_block')}")

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
            
            nt_map = {"Monster": 1, "Elite": 2, "Unknown": 3, "RestSite": 4, "Shop": 5, "Treasure": 6, "Boss": 7}
            map_type = node.get("type")
            assert map_type in nt_map, f"map type: {map_type} not in nt_map"
            map_vec[base_idx + 3] = nt_map[map_type] / 10.0
            
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
                rt_map = {"GoldReward": 1, "Gold": 1, "Card": 2, "CardReward": 2, "Relic": 3, "RelicReward": 3, "PotionReward": 4, "Potion": 4, "Curse": 5}
                reward_type = reward.get("type")
                assert reward_type in rt_map, f"reward type: {reward_type} not in rt_map"
                event_vec[base_idx + 1] = rt_map[reward_type] / 10.0
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
             
    if st_idx in [3, 4]:
        # Feature encoding for card selection (Grid or Hand selection screens)
        cards = state.get("cards", [])
        for i in range(min(len(cards), 10)):
            card = cards[i]
            base_idx = i * 4
            event_vec[base_idx] = 1.0 # Presence flag
            
            # Card ID index
            card_id = card.get("id") or card.get("name", "")
            event_vec[base_idx + 1] = get_card_idx(card_id)
            
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
        "relic_ids": relic_ids,
        "draw_bow": draw_bow,
        "discard_bow": discard_bow,
        "exhaust_bow": exhaust_bow,
        "master_bow": master_bow,
        "map": map_vec,
        "event": event_vec,
        "state_type": np.int32(st_idx),
        "head_type": np.int32(head_idx)
    }

def compute_reward(state, state_type=None):
    """Compute the reward for the current state.
    This is now a final reward: returns 0.0 unless the state is game_over.
    """
    if state_type != "game_over" or reward_tracker.episode_end_recorded:
        return 0.0
    
    # Final reward: Victory or Defeat
    victory = state.get("victory", False)
    
    if victory:
        reward = 5.0
    else:
        reward = -1.0
            
    return reward

def compute_intermediate_reward(state, state_type, action_idx):
    """Compute intermediate reward during a run."""
    intermediate_reward = 0.0
    
    # Current state values
    current_floor = state.get("floor", 0)
    player_data = state.get("player", {}) or {}
    current_hp = int(player_data.get("hp", state.get("hp", 0)) or 0)
    enemies = state.get("enemies", []) or []
    current_enemy_hp = int(sum(e.get("hp", 0) for e in enemies if e.get("hp", 0) > 0 and not e.get("isMinion", False)) or 0)

    # Floor progression reward
    if current_floor > reward_tracker.last_processed_floor:
        # Floor progression reward removed per user request
        
        # Update trackers when floor changes
        reward_tracker.last_processed_floor = current_floor
        reward_tracker.last_player_hp = current_hp
        reward_tracker.last_total_enemy_hp = current_enemy_hp
        reward_tracker.combat_initialized = False # Reset combat init on floor change
        # return intermediate_reward # Allow further processing for other rewards on floor change

    # Combat delta reward (Dense Reward)
    if state_type == "combat":
        # Task 2: Fix combat initialization to avoid start-of-combat penalty
        if not reward_tracker.combat_initialized:
            reward_tracker.initialize_combat(current_hp, current_enemy_hp)
            # return 0.0 # No delta on initialization step
        
        last_hp = reward_tracker.last_player_hp
        last_enemy_hp = reward_tracker.last_total_enemy_hp
        
        # Combat delta reward: Reward damage dealt and HP changes
        # Enemy HP: Only reward damage dealt (don't penalize enemy heals/summons)
        damage_dealt = max(0.0, float(last_enemy_hp - current_enemy_hp))
        # Player HP: Reward both damage taken (penalty) and healing (reward) at the same ratio (0.015)
        hp_delta = float(current_hp - last_hp)
        
        combat_reward = (damage_dealt * 0.002) + (hp_delta * 0.03)
        
        if abs(combat_reward) > 1e-6:
            intermediate_reward += combat_reward
        
        # Track if it was an elite or boss room
        room_type = state.get("room_type", "")
        if room_type == "Elite":
            reward_tracker.was_elite = True
        elif room_type == "Boss":
            reward_tracker.was_boss = True

        # Update trackers for next step
        reward_tracker.last_player_hp = current_hp
        reward_tracker.last_total_enemy_hp = current_enemy_hp

    # Battle Clear Reward
    if reward_tracker.last_state_type == "combat" and (state_type == "rewards" or state_type == "map"):
        intermediate_reward += 0.1 # Battle clear base
        log("Reward for battle clear: +0.1")
        if reward_tracker.was_elite:
            intermediate_reward += 0.1
            log("Extra reward for ELITE defeat: +0.1")
        if reward_tracker.was_boss:
            intermediate_reward += 0.5
            log("Extra reward for BOSS defeat: +0.5")
        reward_tracker.was_elite = False
        reward_tracker.was_boss = False

    # Card / Potion Acquisition Reward (+0.01)
    # Potions Count
    potions = state.get("potions", [])
    current_potion_count = sum(1 for p in potions if p.get("id") != "empty")
    if reward_tracker.last_potion_count != -1:
        if current_potion_count > reward_tracker.last_potion_count:
            if state_type in ["rewards", "shop", "treasure"]: # After battle reward, shop buy, or treasure
                intermediate_reward += 0.01
                log(f"Reward for potion acquisition: +0.01")
        elif current_potion_count < reward_tracker.last_potion_count:
            # 0.001 penalty for using/losing a potion to prevent wasteful use
            intermediate_reward -= 0.001
            log(f"Penalty for potion use: -0.001")
    reward_tracker.last_potion_count = current_potion_count

    # Card Acquisition Count (Using piles in combat or reward list in selection)
    total_cards = -1
    if state_type == "combat":
        # Sum piles as proxy for deck size
        total_cards = (len(state.get("drawPile", [])) + 
                       len(state.get("discardPile", [])) + 
                       len(state.get("exhaustPile", [])) + 
                       len(state.get("hand", [])))
    elif state_type == "grid_selection" or state_type == "card_reward":
        total_cards = len(state.get("cards", []))

    # Reward for card acquisition (only if total_cards is valid and was valid before)
    if reward_tracker.last_total_cards != -1 and total_cards > reward_tracker.last_total_cards:
        # Note: In grid_selection, total_cards might fluctuate based on screen. 
        # But if it increases, it's likely an acquisition (e.g. from RewardCard)
        # We only give this if we are not in combat or if it's a permanent addition.
        # Actually, let's keep it simple: if it increases, reward it.
        if state_type != "combat" or reward_tracker.last_state_type == "combat": # Transition or outside
            intermediate_reward += 0.01
            log(f"Reward for card acquisition: +0.01")
    reward_tracker.last_total_cards = total_cards

    # Card Upgrade Reward (+0.01)
    if state_type == "grid_selection" and state.get("subtype") == "NDeckUpgradeSelectScreen":
        cards = state.get("cards", [])
        upgraded_count = sum(1 for c in cards if c.get("upgraded"))
        if reward_tracker.last_upgraded_count != -1 and upgraded_count > reward_tracker.last_upgraded_count:
            intermediate_reward += 0.01
            log("Reward for card upgrade: +0.01")
        reward_tracker.last_upgraded_count = upgraded_count
    else:
        reward_tracker.last_upgraded_count = -1

    reward_tracker.last_state_type = state_type
        
    return intermediate_reward

def needs_target(card):
    """Returns True if the card requires a specific enemy or ally target."""
    target_type = card.get("targetType", "None")
    # Cards that target 'AllEnemies' or 'RandomEnemy' don't need a target index in the action.
    # Cards that target 'Self' or 'None' also don't need a target index.
    return "Enemy" in target_type or "Single" in target_type or "Ally" in target_type or "Player" in target_type

def get_action_mask(state, masked_reward_indices=None):
    do_deferred_imports()
    assert np is not None
    assert jnp is not None
    mask = np.zeros(100, dtype=bool)
    state_type = state.get("type", "unknown")
    
    if state_type == "combat":
        hand = state.get("hand") or []
        enemies = state.get("enemies", []) or []
        num_enemies = len(enemies)
        
        actions_disabled = state.get("actions_disabled", False)
        
        if not actions_disabled:
            for i in range(min(len(hand), 10)):
                card = hand[i]
                if card.get("isPlayable"):
                    if needs_target(card):
                        # Task 3: Fix target masking for gaps in indices
                        for t in range(min(len(enemies), 5)):
                            enemy = enemies[t]
                            if enemy.get("hp", 0) > 0:
                                mask[i * 5 + t] = True
                    else:
                        # Self, AllEnemies, None, or Random target cards use target_idx 0 in the action space
                        mask[i * 5] = True
        else:
            log("Python-Bridge: actions_disabled detected in combat. Masking cards.")
        
        # 50-74: Potions (up to 5 potions * 5 targets)
        if not actions_disabled:
            potions = state.get("potions", [])
            for i in range(min(len(potions), 5)):
                potion = potions[i]
                if potion.get("canUse", False):
                    target_type = potion.get("targetType", "None")
                    if "Enemy" in target_type or "Single" in target_type or "Ally" in target_type or "Player" in target_type:
                        for t in range(min(len(enemies), 5)):
                            if enemies[t].get("hp", 0) > 0:
                                mask[50 + i * 5 + t] = True
                    else:
                        mask[50 + i * 5] = True
        else:
            log("Python-Bridge: actions_disabled detected in combat. Masking potions.")
        
        # 75: End Turn (Only if enemies are present and actions are not disabled)
        if not actions_disabled and enemies:
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
        if route_mode and next_nodes:
            # Force the first available node
            mask[0] = True
            # log(f"Python-Bridge: route_mode enabled. Masking all but the first map node (index 0).")
        else:
            for i in range(min(len(next_nodes), 10)):
                mask[i] = True
            
    elif state_type == "event":
        options = state.get("options", [])
        for i in range(min(len(options), 10)):
            if not options[i].get("is_locked"):
                mask[i] = True
        
        if state.get("can_proceed"):
            mask[86] = True # Proceed
                
    elif state_type == "rest_site":
        options = state.get("options", [])
        for i in range(min(len(options), 5)):
            if options[i].get("is_enabled"):
                mask[i] = True
        if state.get("can_proceed"):
            mask[86] = True # Proceed

    elif state_type in ["grid_selection", "hand_selection"]:
        cards = state.get("cards", [])
        is_confirming = state.get("is_confirming", False)
        if is_confirming:
            # In confirmation phase, only allow confirming the selection
            mask[90] = True  # confirm_selection
        else:
            for i in range(min(len(cards), 20)):
                # Mask out cards that are already selected to avoid infinite toggle loops
                if not cards[i].get("selected", False):
                    mask[i] = True
            # can_skip or can_proceed allows skipping/confirming (index -1 maps to action 90 for grid)
            if state.get("can_skip") or state.get("can_proceed"):
                mask[90] = True  # skip / confirm
        
    elif state_type == "shop":
        items = state.get("items", [])
        for i in range(min(len(items), 30)):
            if items[i].get("canAfford"):
                mask[i] = True
        # Always allow proceed (leave shop without buying) to avoid stalling
        mask[86] = True
        
    elif state_type == "treasure":
        if state.get("has_chest"):
            mask[91] = True # Open Chest
        if state.get("can_proceed"):
            mask[86] = True # Proceed
            
    elif state_type == "treasure_relics":
        relics = state.get("relics", [])
        for i in range(min(len(relics), 5)):
            mask[i] = True
        
    elif state_type == "card_reward":
        cards = state.get("cards", [])
        for i in range(min(len(cards), 5)):
            mask[i] = True
        buttons = state.get("buttons", [])
        for i in range(min(len(buttons), 5)):
            mask[10 + i] = True

    elif state_type == "game_over":
        mask[86] = True # Proceed
        mask[87] = True # Return to Main Menu

    # 94-98: Discard Potion (indices 0-4)
    # This should be available in combat or potentially other states where potions are visible
    potions = state.get("potions", [])
    for i in range(min(len(potions), 5)):
        if potions[i].get("id") != "empty":
            mask[94 + i] = True

    # Action 99 (Wait) is disabled since waiting is handled in C#
    mask[99] = False
    
    if not np.any(mask):
        # If no actions are possible, we MUST allow WAIT to let the game progress (e.g., during animations)
        # and prevent the AI from picking a random (masked) action that might lead to a crash or stall.
        mask[99] = True
        summary = {
            "floor": state.get("floor"),
            "hp": state.get("player", {}).get("hp") if isinstance(state.get("player"), dict) else None,
            "hand_count": len(state.get("hand", [])) if isinstance(state.get("hand", list)) else 0,
            "enemy_count": len(state.get("enemies", [])) if isinstance(state.get("enemies", list)) else 0,
            "can_proceed": state.get("can_proceed")
        }
        log(f"WARNING: No valid actions in mask for state {state_type}. Enabling WAIT (99). State Summary: {json.dumps(summary)}")
        
    return mask

def check_commands():
    """Poll for background commands (like start_game) without triggering AI inference."""
    global command_queue
    # log(f"check_commands polled (queue size: {command_queue.qsize()})")
    if not command_queue.empty():
        cmd = command_queue.get_nowait()
        log(f"check_commands: processing command {cmd}")
        if ":" in cmd:
            base_cmd, extra = cmd.split(":", 1)
            return json.dumps({
                "action": "command",
                "command": base_cmd,
                "seed": extra
            })
        return json.dumps({"action": "command", "command": cmd})
    return json.dumps({"action": "wait"})

def predict_action(state_json):
    global can_continue_status
    global is_restoring
    t0 = time.time()
    do_deferred_imports()
    # Explicitly refer to the global jax module after deferred imports
    import jax as jax_local
    
    t_imports = time.time() - t0
    
    global command_queue, learning_active, current_seed, current_trajectory, learner, rng_key, predict_key, last_activity_time, deferred_chunk, history
    
    try:
        t_json_start = time.time()
        state = json.loads(state_json)
        if "can_continue" in state:
            can_continue_status = state["can_continue"]
        t_json = time.time() - t_json_start
        state_type = state.get("type", "unknown")
        
        # Update last activity time
        if state_type not in ["none", "main_menu", "unknown"]:
            last_activity_time = time.time()
        
        # Track episode end for mean last floor calculation
        if state_type == "game_over":
            # Use a flag to record only once per game_over screen. 
            # Reset the flag when we see any gameplay state.
            if not reward_tracker.episode_end_recorded:
                floor = state.get("floor", 1)
                terminal_reward = compute_reward(state, state_type)
                total_reward = reward_tracker.session_cumulative_reward + terminal_reward
                log(f"Episode end detected at floor {floor}, final reward {total_reward:.2f}. Recording...")
                if training_worker:
                    training_worker.record_game_over(floor, total_reward)
                reward_tracker.episode_end_recorded = True
                
                # Flush trajectory if it exists
                if current_trajectory:
                    log(f"Attributing terminal reward {terminal_reward:.2f} to the last step of the episode.")
                    current_trajectory[-1]["rew"] += terminal_reward
                    current_trajectory[-1]["done"] = 1.0
                    
                    log(f"Flushing terminal trajectory of length {len(current_trajectory)}")
                    experience_queue.put(list(current_trajectory))
                    current_trajectory = []
                
                # Apply penalty to session cumulative reward for UI visibility
                reward_tracker.session_cumulative_reward += terminal_reward
                log(f"Applied terminal penalty {terminal_reward:.2f} to session_cumulative_reward. New total: {reward_tracker.session_cumulative_reward:.2f}")

                # Reset reward for next episode (moved to next run start or main menu to allow UI to see final result)
                # reward_tracker.reset_for_next_episode()
            
            # Clear history on episode end
            history = []
            
            # Flush any deferred chunk (terminal)
            if deferred_chunk:
                log(f"Flushing terminal deferred chunk of length {len(deferred_chunk['steps'])}")
                experience_queue.put(deferred_chunk)
                deferred_chunk = None

        elif state_type in ["combat", "map", "event", "rest_site", "shop", "treasure"]:
            reward_tracker.episode_end_recorded = False
            reward_tracker.skipped_reward_indices = set()
            reward_tracker.last_reward_floor = state.get("floor", -1)
        elif state_type == "rewards":
            reward_tracker.episode_end_recorded = False
            if reward_tracker.last_reward_floor != state.get("floor"):
                reward_tracker.skipped_reward_indices = set()
                reward_tracker.last_reward_floor = state.get("floor")
        elif state_type in ["main_menu", "none"]:
            reward_tracker.reset_for_new_run()
        if not state_json:
            return json.dumps({"action": "wait"})

        log(f"predict_action called. state_type: {state_type}")
        
        # Debug: write last state to a local file for monitoring
        try:
            last_state_path = os.path.join(LOG_DIR, "rnad_last_state.json")
            display_state = json.loads(state_json)
            # If we are in the middle of a restoration, tell the supervisor to wait
            if is_restoring:
                display_state["type"] = "restoration_pending"
            
            with open(last_state_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(display_state))
        except:
            pass

        if state_type in ["none", "main_menu", "unknown", "combat_waiting"]:
            if state_type == "main_menu":
                is_restoring = False # Reset restoration flag when we reach main menu
            return json.dumps({"action": "wait"})

        if not initialization_event.is_set():
            log("[Python] Waiting for model initialization...")
            if not initialization_event.wait(timeout=300):
                log("[Python] Timeout waiting for model initialization!")
                return json.dumps({"action": "error", "message": "initialization timeout"})
            log("[Python] Model initialization complete. Proceeding with predict_action.")

        state = json.loads(state_json)
        state_type = state.get("type", "unknown")
        
        # Simulator Validation (Validate the outcome of the PREVIOUS action)
        validator.validate(state_json)
        
        state_dict = encode_state(state)
        
        # Calculate Action Mask
        masked_rewards = reward_tracker.skipped_reward_indices
        mask = get_action_mask(state, masked_reward_indices=masked_rewards)
        do_deferred_imports()
        assert np is not None
        assert jnp is not None
        mask_jnp = jnp.array(mask)
        
        # Check if we have a deferred chunk waiting for this state to bootstrap
        if deferred_chunk:
            log(f"Pushing deferred chunk with {len(deferred_chunk['steps'])} steps using current state for bootstrapping.")
            deferred_chunk["next_step"] = {"obs": state_dict, "mask": mask_jnp}
            experience_queue.put(deferred_chunk)
            deferred_chunk = None

        # Update history for temporal context
        history.append((state_dict, mask_jnp))
        if config and len(history) > config.seq_len:
            history.pop(0)

        # Preparing dictionary with fixed T (seq_len) for inference to avoid re-compilation
        T_actual = len(history)
        T_limit = config.seq_len if config else 8
        
        batched_state = {}
        for k in state_dict.keys():
            # Extract real data from history
            real_data = jnp.stack([h[0][k] for h in history], axis=0)
            # Pad with zeros to fixed T_limit at the end
            # Shape for padding: (T_limit - T_actual, ...)
            pad_shape = (T_limit - T_actual,) + real_data.shape[1:]
            padded_data = jnp.concatenate([real_data, jnp.zeros(pad_shape, dtype=real_data.dtype)], axis=0)
            batched_state[k] = padded_data[:, None, ...] # (T_limit, 1, ...)
        
        real_mask = jnp.stack([h[1] for h in history], axis=0)
        pad_mask_shape = (T_limit - T_actual,) + real_mask.shape[1:]
        padded_mask = jnp.concatenate([real_mask, jnp.zeros(pad_mask_shape, dtype=real_mask.dtype)], axis=0)
        batched_mask = padded_mask[:, None, :] # (T_limit, 1, num_actions)
        
        # Inference mode toggle
        inference_mode = os.environ.get("RNAD_INFERENCE_MODE") == "true"
        
        # Inference
        t_inf_start = time.time()
        predict_key, rng_key = jax_local.random.split(rng_key)
        
        if learner is None:
            log("[Python] CRITICAL: learner is None in predict_action!")
            return json.dumps({"action": "error", "message": "learner is None"})
            
        if learner.params is None:
            log("[Python] CRITICAL: learner.params is None in predict_action!")
            # Emergency fix: if we reached here after wait, something is wrong with the loaded model
            return json.dumps({"action": "error", "message": "learner.params is None after initialization"})
        
        # Aggressive debug: check if hg_proj is in params (can be prefixed like transformer_net/hg_proj)
        hg_proj_exists = any('hg_proj' in k for k in learner.params.keys())
        if not hg_proj_exists:
            log(f"[Python] CRITICAL: hg_proj MISSING from learner.params! Keys: {list(learner.params.keys())}")
        elif 'hg_proj' not in learner.params:
            # It exists but is prefixed, which is expected. 
            # We skip the critical log but could log a quiet confirmation if we wanted.
            pass
        
        # Pass is_training flag to the network
        is_training = not inference_mode
        
        if _predict_step is None:
            # Fallback if jit failed
            logits_seq, value_seq = learner.network.apply(learner.params, predict_key, batched_state, batched_mask.astype(jnp.float32), is_training=is_training)
        else:
            # Note: _predict_step is JITed in load_model. We need to make sure it handles is_training.
            # However, _predict_step currently doesn't take is_training as an argument.
            # We'll need to update load_model as well.
            logits_seq, value_seq = _predict_step(learner.params, predict_key, batched_state, batched_mask.astype(jnp.float32), is_training)
        t_inference = time.time() - t_inf_start
        
        # Rollout: we only care about the actual latest step in the sequence (index T_actual - 1)
        logits = logits_seq[T_actual - 1 : T_actual] # (1, B, num_actions)
        value = value_seq[T_actual - 1 : T_actual]  # (1, B)
        
        # Probs are calculated from logits which are already masked in the network
        probs = jax_local.nn.softmax(logits, axis=-1)[0, 0]

        # --- Retry Probability Adjustment ---
        # Reduce probability for actions taken in past trials of the same retry point
        penalized_actions = backup_manager.get_penalized_actions()
        if penalized_actions:
            for act in penalized_actions:
                if act < len(probs) and mask[act]:
                    orig_p = float(probs[act])
                    probs = probs.at[act].multiply(0.75)
                    new_p = float(probs[act])
                    log(f"[Python] Retry Probability Adjustment: Penalizing action {act} (orig={orig_p:.4f}, new={new_p:.4f})")
            
            # Renormalize
            sum_p = jnp.sum(probs)
            if sum_p > 1e-9:
                probs = probs / sum_p
            else:
                log("[Python] WARNING: All actions penalized to zero in Retry Adjustment! Resetting probs.")
                probs = jax_local.nn.softmax(logits, axis=-1)[0, 0]
        
        # --- Skip Override Logic ---
        sampling_mask = mask.copy()
        if state_type == "card_reward":
            buttons = state.get("buttons", [])
            room_type = state.get("room_type", "")
            if mask_card_skip and room_type in ["Monster", "Elite", "Boss"]:
                for i in range(min(len(buttons), 5)):
                    btn_name = buttons[i].get("name", "").lower()
                    if "skip" in btn_name or "スキップ" in btn_name:
                        if sampling_mask[10 + i]:
                            sampling_mask[10 + i] = False
                            log(f"[Python] SKIP OVERRIDE: Zeroing prob for Skip button at index {10+i} (original prob: {probs[10+i]:.4f})")
                
                # Re-calculate probs based on the sampling_mask
                # This ensures "selecting from others according to their relative probabilities"
                probs = probs * sampling_mask
                sum_p = jnp.sum(probs)
                if sum_p > 1e-9:
                    probs = probs / sum_p
                else:
                    log("[Python] WARNING: All actions masked during Skip override! Reverting to original mask.")
                    sampling_mask = mask.copy()
                    probs = jax_local.nn.softmax(logits, axis=-1)[0, 0]
        # ---------------------------
        # --- Turn End Override Logic ---
        # If energy > 0 during training, scale End Turn (75) prob by 0.01 to encourage exploration
        if not inference_mode and state_type == "combat":
            player_info = state.get("player", {})
            energy = player_info.get("energy", 0)
            if energy > 0 and mask[75]:
                # Action 75 is End Turn
                probs = probs.at[75].multiply(0.01)
                sum_p = jnp.sum(probs)
                if sum_p > 1e-9:
                    probs = probs / sum_p
                log(f"[Python] TURN END OVERRIDE: Scaling prob for End Turn (75) by 0.01 due to energy={energy}. New prob: {probs[75]:.4f}")
        # ---------------------------
        # --- Search Features ---
        # Lethal Search: Harness to ensure we don't miss a deterministic kill-all sequence.
        # Multiple Move: Full tree search with value-head evaluation for best next state.
        lethal_search_mode = os.environ.get("RNAD_LETHAL_SEARCH", "true") == "true"
        multiple_move_mode = os.environ.get("RNAD_MULTIPLE_MOVE", "false") == "true"
        search_action_idx = None

        if (lethal_search_mode or multiple_move_mode) and state_type == "combat" and battle_simulator is not None:
            log(f"[Python] Combat Search: START (lethal={lethal_search_mode}, multi={multiple_move_mode})")
            try:
                sim_json = validator.to_simulator_json(state)
                sim = battle_simulator.Simulator.from_json(sim_json)
                sim_manager.init_simulator(sim)
                num_res = sim.enumerate_final_states()
                log(f"[Python] Combat Search: Enumerate states finished. Found {num_res} raw states.")
                if num_res > 0:
                    all_results = sim_manager.read_results()
                    # Filter for deterministic sequences (those that end with End Turn action 75)
                    results = [r for r in all_results if r["actions"] and r["actions"][-1] == 75]
                    if len(results) == 0:
                        log(f"[Python] Combat Search: No deterministic sequences found (raw={num_res}).")
                    
                    # 1. Lethal Search (Always prioritize kill-all if enabled)
                    ka_seqs = []
                    for res in results:
                        is_ka = True
                        for prob, tens in res["outcomes"]:
                            # tens[622] is Enemy 0 Alive flag. We need to check all 5 possible enemy slots.
                            # Each enemy has 16 features. Offsets: 622, 638, 654, 670, 686
                            enemy_alive_flags = [tens[622 + i * 16] for i in range(5)]
                            if any(flag > 0.5 for flag in enemy_alive_flags):
                                is_ka = False
                                break
                        if is_ka: ka_seqs.append(res)
                    
                    if ka_seqs and lethal_search_mode:
                        # Prioritize Kill-all by player HP (tensor[2])
                        best_ka = max(ka_seqs, key=lambda r: sum(p * t[2] for p, t in r["outcomes"]))
                        search_action_idx = best_ka["actions"][0]
                        orig_p = float(probs[search_action_idx])
                        log(f"[Python] Lethal Search: KILL-ALL sequence FOUND! ({len(ka_seqs)} paths, best_ev={sum(p * t[2] for p, t in best_ka['outcomes']):.2f}). Selecting Action={search_action_idx} (Policy Prob={orig_p:.4f})")
                    
                    elif multiple_move_mode and learner and learner.params:
                        if len(results) > 0:
                            log(f"[Python] Multiple Move: No kill-all sequence found in {len(results)} deterministic paths. Evaluating via value-head.")
                        
                        # ... existing evaluation logic ...
                        eval_res = results[:32] if len(results) > 32 else results
                        outcomes_to_eval = []
                        for ridx, r in enumerate(eval_res):
                            for oidx, (p, t) in enumerate(r["outcomes"]):
                                outcomes_to_eval.append((ridx, oidx, p, t))
                        
                        if outcomes_to_eval:
                            t_batch = np.stack([o[3] for o in outcomes_to_eval])
                            # BOW features start at offset 896. Each pile has size VOCAB_SIZE (611).
                            b_off = 896
                            v_sz = VOCAB_SIZE
                            obs_batch = {
                                "global": jnp.array(t_batch[:, :512])[:, None, :],
                                "combat": jnp.array(t_batch[:, 512:b_off])[:, None, :],
                                "draw_bow": jnp.array(t_batch[:, b_off : b_off + v_sz])[:, None, :],
                                "discard_bow": jnp.array(t_batch[:, b_off + v_sz : b_off + 2*v_sz])[:, None, :],
                                "exhaust_bow": jnp.array(t_batch[:, b_off + 2*v_sz : b_off + 3*v_sz])[:, None, :],
                                "master_bow": jnp.array(t_batch[:, b_off + 3*v_sz : b_off + 4*v_sz])[:, None, :],
                                "state_type": jnp.array(t_batch[:, b_off + 4*v_sz], dtype=jnp.int32)[:, None],
                                "head_type": jnp.array(t_batch[:, b_off + 4*v_sz + 1], dtype=jnp.int32)[:, None],
                            }
                            m_batch = jnp.ones((len(t_batch), 1, 100))
                            s_key, predict_key = jax_local.random.split(predict_key)
                            _, v_out = learner.network.apply(learner.params, s_key, obs_batch, m_batch, is_training=False)
                            
                            r_vals = np.zeros(len(eval_res))
                            for idx, (ridx, oidx, p, t) in enumerate(outcomes_to_eval):
                                r_vals[ridx] += p * v_out[idx, 0].item()
                            
                            best_idx = np.argmax(r_vals)
                            search_action_idx = eval_res[best_idx]["actions"][0]
                            orig_p = float(probs[search_action_idx])
                            log(f"[Python] Search: Best-Value sequence identified. Action={search_action_idx} (EV={r_vals[best_idx]:.4f}, Policy Prob={orig_p:.4f})")
                
                if search_action_idx is None:
                    log(f"[Python] Combat Search: No optimal sequence identified by search. Falling back to policy.")
                
                log(f"[Python] Combat Search: COMPLETED.")
            except Exception as e:
                log(f"[Python] Multiple Move Search ERROR: {e}")
                traceback.print_exc()
        
        # Sample or Argmax action
        is_search_override = False
        if search_action_idx is not None and mask[search_action_idx]:
            action_idx = search_action_idx
            is_search_override = True
            log(f"[Python] Using SEARCH-selected action: {action_idx} (Policy prob={probs[action_idx]:.4f})")
        elif inference_mode:
            # Highest probability selection (Greedy)
            action_idx = jnp.argmax(probs).item()
            log(f"[Python] Inference mode (Greedy selection): index={action_idx}, prob={probs[action_idx]:.4f}")
        else:
            # Stochastic selection (Sampling)
            rng_key, subkey = jax_local.random.split(rng_key)
            # Use choice with adjusted probs
            action_idx = jax_local.random.choice(subkey, jnp.arange(len(probs)), p=probs).item()
            
        # Store state for next validation if we're in combat
        # Skip storing if it's the End Turn action (75) to avoid false discrepancies next time
        if state_type == "combat" and action_idx != 75:
            validator.last_state = state
            validator.last_action_idx = action_idx
            
        # Override for Map transition (Floor Start)
        if state_type == "map" and state.get("current_pos") is None:
            next_nodes = state.get("next_nodes", [])
            if next_nodes:
                log(f"[Python] Map transition detected (current_pos is None). Auto-selecting first node (index 0).")
                action_idx = 0
            
        # Record this action in the trial history for future diversification
        backup_manager.record_action(action_idx)

        log_prob = jnp.log(jnp.maximum(probs[action_idx], 1e-9)).item()
        
        # ▼ [DEBUG LOGGING]
        try:
            # Prepare a concise summary of the state
            player_info = state.get("player", {})
            state_summary = {
                "floor": state.get("floor"),
                "hp": player_info.get("hp") if player_info else state.get("hp"),
                "energy": player_info.get("energy") if player_info else state.get("energy"),
                "gold": state.get("gold")
            }
            
            # Format probabilities for logging (all 100)
            probs_list = probs.tolist()
            mask_list = mask.tolist()
            
            log_decision(f"--- AI Decision Log ---")
            log_decision(f"State Type: {state_type}")
            log_decision(f"State Summary: {json.dumps(state_summary)}")
            log_decision(f"Action Mask (first 20): {mask_list[:20]} ...")
            log_decision(f"Selected Action Index: {action_idx}")
            
            # Identify action for logging
            ident_action = "wait"
            if state_type == "combat":
                if action_idx < 50:
                    card_idx = action_idx // 5
                    if card_idx < len(state.get("hand", [])):
                        card = state['hand'][card_idx]
                        ident_action = f"play_card:{card.get('id')}"
                        if needs_target(card):
                            target_idx = action_idx % 5
                            enemies = state.get("enemies", [])
                            if target_idx < len(enemies):
                                ident_action += f" on {enemies[target_idx].get('name')}"
                            else:
                                ident_action += f" on target_{target_idx}"
                elif 50 <= action_idx < 75:
                    pots = state.get("potions", [])
                    potion_idx = (action_idx - 50) // 5
                    if potion_idx < len(pots):
                        pot = pots[potion_idx]
                        ident_action = f"use_potion:{pot.get('id')}"
                        target_type = pot.get("targetType", "None")
                        if "Enemy" in target_type or "Single" in target_type or "Ally" in target_type or "Player" in target_type:
                            target_idx = (action_idx - 50) % 5
                            enemies = state.get("enemies", [])
                            if target_idx < len(enemies):
                                ident_action += f" on {enemies[target_idx].get('name')}"
                            else:
                                ident_action += f" on target_{target_idx}"
                elif action_idx == 75: ident_action = "end_turn"
            
            log_decision(f"Selected Action ID: {ident_action}")
            
            # Detailed probabilities log
            for i in range(0, 100, 10):
                chunk_probs = [f"{p:.4f}" for p in probs_list[i:i+10]]
                log_decision(f"Probs[{i:02d}-{i+9:02d}]: {' '.join(chunk_probs)}")
            
            log_decision(f"Value Estimate: {value[0].item():.4f}")
            log_decision(f"-----------------------")
        except Exception as log_e:
            log(f"Error during debug logging: {log_e}")
        # ▲ [DEBUG LOGGING]

        # Log if the model's preferred (original top) action was masked.
        # Note: Since the network now applies masking, we'd need to check raw logits before masking
        # to see what it "originally" wanted, but the network internally hides that.
        original_top_action = jnp.argmax(logits).item()
        
        action = {"action": "wait"} # Default
        
        if action_idx == 99:
            pass # already wait
        
        elif state_type == "combat":
            hand = state.get("hand") or []
            potions = state.get("potions", [])
            
            if action_idx < 50:
                card_idx = action_idx // 5
                target_idx = action_idx % 5
                if card_idx < len(hand):
                    card = hand[card_idx]
                    action = {"action": "play_card", "card_id": card.get("id"), "card_index": card_idx}
                    if needs_target(card):
                        action["target_index"] = target_idx
            elif 50 <= action_idx < 75:
                potion_linear_idx = action_idx - 50
                potion_idx = potion_linear_idx // 5
                target_idx = potion_linear_idx % 5
                if potion_idx < len(potions):
                    potion = potions[potion_idx]
                    action = {"action": "use_potion", "index": potion_idx}
                    target_type = potion.get("targetType", "None")
                    if "Enemy" in target_type or "Single" in target_type or "Ally" in target_type or "Player" in target_type:
                        action["target_index"] = target_idx
            elif action_idx == 75:
                action = {"action": "end_turn"}
            elif action_idx == 86:
                action = {"action": "proceed"}
        
        elif state_type == "rewards":
            rewards = state.get("rewards", [])
            if 76 <= action_idx < 86:
                reward_idx = action_idx - 76
                if reward_idx < len(rewards):
                    action = {"action": "select_reward", "index": reward_idx}
            elif action_idx == 86:
                action = {"action": "proceed"}
        
        elif state_type == "map":
            next_nodes = state.get("next_nodes", [])
            if action_idx < len(next_nodes):
                node = next_nodes[action_idx]
                action = {"action": "select_map_node", "row": node["row"], "col": node["col"]}
        
        elif state_type == "event":
            options = state.get("options", [])
            if action_idx < len(options):
                action = {"action": "select_event_option", "index": options[action_idx].get("index")}
        
        elif state_type == "rest_site":
            options = state.get("options", [])
            if action_idx < len(options):
                action = {"action": "select_rest_site_option", "index": options[action_idx].get("index")}
            elif action_idx == 86:
                action = {"action": "proceed"}
        
        elif state_type == "shop":
            items = state.get("items", [])
            if action_idx < len(items):
                action = {"action": "buy_item", "index": items[action_idx].get("index")}
            elif action_idx == 86:
                action = {"action": "shop_proceed"}
        
        elif state_type == "treasure":
            if action_idx == 91:
                action = {"action": "open_chest"}
            elif action_idx == 86:
                action = {"action": "proceed"}
        
        elif state_type == "treasure_relics":
            relics = state.get("relics", [])
            if action_idx < len(relics):
                action = {"action": "select_treasure_relic", "index": relics[action_idx].get("index")}
        
        elif state_type == "card_reward":
            cards = state.get("cards", [])
            buttons = state.get("buttons", [])
            if action_idx < 5:
                if action_idx < len(cards):
                    action = {"action": "select_reward_card", "index": cards[action_idx].get("index")}
            elif 10 <= action_idx < 15:
                btn_idx = action_idx - 10
                if btn_idx < len(buttons):
                    action = {"action": "click_reward_button", "index": buttons[btn_idx].get("index")}
        
        elif state_type in ["grid_selection", "hand_selection"]:
            cards = state.get("cards", [])
            if action_idx < 20:
                if action_idx < len(cards):
                    if state_type == "grid_selection":
                        action = {"action": "select_grid_card", "index": cards[action_idx].get("index")}
                    else:
                        action = {"action": "select_hand_card", "index": cards[action_idx].get("index")}
            elif action_idx == 90:
                action = {"action": "confirm_selection"}
        
        elif 94 <= action_idx < 99:
            potion_idx = action_idx - 94
            action = {"action": "discard_potion", "index": potion_idx}
        
        elif state_type == "game_over":
            if action_idx == 86:
                action = {"action": "proceed"}
            elif action_idx == 87:
                # User requested: Trigger restore on Game Over when returning to main menu
                if not is_restoring:
                    log("[Python] Game Over: Initiating restoration sequence...")
                    if trigger_restore():
                        is_restoring = True
                        log("[Python] Game Over: Restoration successful. Returning to main menu.")
                    else:
                        log("[Python] Game Over: Restoration failed (no backups). Returning to main menu for fresh start.")
                action = {"action": "return_to_main_menu"}

        # Trajectory collection
        if learning_active:
            # ▼修正: 有効な状態のみ記録し、waitアクション(99)は除外する
            if action_idx != 99 and (state_type in VALID_TRAJECTORY_STATES or state_type == "game_over"):
                base_reward = compute_reward(state, state_type)
                
                # Intermediate rewards (floor progression and combat end_turn)
                intermediate_reward = compute_intermediate_reward(state, state_type, action_idx)
 
                reward = base_reward + intermediate_reward
                
                # Accumulate session reward
                reward_tracker.session_cumulative_reward += reward
 
                # Existing experience collection for TrainingWorker
                if state_type in VALID_TRAJECTORY_STATES:
                    current_trajectory.append({
                        "obs": state_dict,
                        "act": int(action_idx),
                        "rew": float(reward),
                        "mask": sampling_mask.astype(np.float32),
                        "log_prob": float(log_prob),
                        "probs": probs.tolist() if hasattr(probs, "tolist") else list(probs),
                        "logits": logits[0, 0].tolist() if hasattr(logits, "tolist") else list(logits[0, 0]),
                        "predicted_v": float(value.item()),
                        "done": 0.0
                    })
                
                # NEW: Raw trajectory logging for offline learning
                terminal = (state_type == "game_over")
                raw_logger.log_step(state_json, action_idx, probs, mask, reward, log_prob, predicted_v=value.item(), logits=logits[0, 0], terminal=terminal, is_search=is_search_override)

                if config and len(current_trajectory) >= config.unroll_length:
                    # Defer flushing until we see the next state for bootstrapping
                    log(f"Trajectory reached unroll_length ({config.unroll_length}). Moving to deferred_chunk.")
                    deferred_chunk = {"steps": list(current_trajectory)}
                    current_trajectory = []
 
        # If we chose to skip a card reward, remember this to mask it in next rewards screen call
        if action.get("action") == "click_reward_button":
             if action.get("index") is not None:
                 buttons = state.get("buttons", [])
                 if action["index"] < len(buttons):
                     btn_name = buttons[action["index"]].get("name", "").lower()
                     if "skip" in btn_name or "remov" in btn_name or "dismiss" in btn_name: 
                         if reward_tracker.last_selected_reward_idx is not None:
                             log(f"Detected SKIP in card reward. Masking reward index {reward_tracker.last_selected_reward_idx} for floor {state.get('floor')}")
                             last_idx = reward_tracker.last_selected_reward_idx
                             if isinstance(last_idx, (int, float)):
                                 reward_tracker.skipped_reward_indices.add(int(last_idx))
 
        if action.get("action") == "select_reward":
            idx = action.get("index")
            reward_tracker.last_selected_reward_idx = int(idx) if idx is not None else None
 
        res = json.dumps(action)
        t_total = time.time() - t0
        t_inference_ms = (t_inference * 1000) if 't_inference' in locals() else 0
        log(f"[PERF] predict_action({state_type}) took {t_total*1000:.2f}ms (inf:{t_inference_ms:.2f}ms). Action: {res}")
        return res
            
    except Exception as e:
        print(f"[Python] CRITICAL ERROR in predict_action: {e}")
        traceback.print_exc()
        return json.dumps({"action": "error", "message": str(e)})

class CommandHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global learning_active, command_queue, current_seed, current_trajectory, experience_queue
        parsed_path = urlparse(self.path)
        log(f"[HTTP] GET {parsed_path.path}")
        
        if parsed_path.path == "/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            
            # Combine experience_queue, training_worker.batch_buffer, and deferred_chunk for more accurate queue reporting
            total_queue_size = experience_queue.qsize()
            if training_worker:
                total_queue_size += len(training_worker.batch_buffer)
            
            # Include deferred_chunk in queue size if it exists, as it's almost ready to be pushed
            if deferred_chunk:
                total_queue_size += 1
                
            self.wfile.write(json.dumps({
                "learning_active": learning_active,
                "can_continue": can_continue_status,
                "queue_size": total_queue_size,
                "has_deferred": deferred_chunk is not None,
                "traj_size": len(current_trajectory),
                "unroll_length": config.unroll_length if config else 0,
                "batch_size": config.batch_size if config else 0,
                "step_count": training_worker.step_count if training_worker else 0,
                "worker_error": training_worker.last_error if training_worker else None,
                "is_updating": training_worker.is_updating if training_worker else False,
                "update_progress": training_worker.update_progress if training_worker else 0,
                "update_total": training_worker.update_total if training_worker else 0,
                "initialized": initialized,
                "last_activity_time": last_activity_time
            }).encode())
            
        elif parsed_path.path == "/state":
            last_state_path = os.path.join(LOG_DIR, "rnad_last_state.json")
            state = {}
            if os.path.exists(last_state_path):
                with open(last_state_path, "r") as f:
                    try:
                        state = json.load(f)
                    except:
                        pass
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(state).encode())
            
        elif parsed_path.path == "/flush_trajectory":
            if current_trajectory:
                log(f"Manual flush: Moving trajectory of length {len(current_trajectory)} to experience_queue.")
                experience_queue.put(list(current_trajectory))
                current_trajectory = []
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "flushed"
            }).encode())

        elif parsed_path.path == "/offline_train":
            if not offline_enabled:
                log("WARNING: /offline_train called but --offline flag was not provided at launch.")
                self.send_response(403)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "forbidden", "message": "Offline training not enabled via --offline flag"}).encode())
                return

            if training_worker:
                def run_offline():
                    try:
                        training_worker.perform_offline_training()
                    except Exception as e:
                        log(f"Error in offline training thread: {e}")
                        traceback.print_exc()

                threading.Thread(target=run_offline, daemon=True).start()
                
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "offline_training_started"}).encode())
            else:
                log("WARNING: /offline_train called but TrainingWorker not initialized yet.")
                self.send_response(503) # Service Unavailable
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": "TrainingWorker not initialized"}).encode())

        elif parsed_path.path == "/save_trajectory":
            # Save current_trajectory, experience_queue, and batch_buffer to disk
            traj_checkpoint_path = os.path.join(LOG_DIR, "trajectory_checkpoint.pkl")
            
            if current_trajectory:
                log(f"/save_trajectory: Flushing trajectory of length {len(current_trajectory)} to experience_queue before saving.")
                experience_queue.put(list(current_trajectory))
                current_trajectory = []

            # Flush raw logger as well
            if raw_logger:
                raw_logger.flush()

            # Safely capture current state
            with experience_queue.mutex:
                queue_snapshot = list(experience_queue.queue)
            
            buffer_snapshot = []
            if training_worker:
                with training_worker.buffer_lock:
                    buffer_snapshot = list(training_worker.batch_buffer)
            
            data_to_save = {
                "current_trajectory": list(current_trajectory),
                "experience_queue": queue_snapshot,
                "batch_buffer": buffer_snapshot,
                "learner_params": learner.params if learner else None,
                "learner_fixed_params": learner.fixed_params if learner else None,
                "learner_opt_state": learner.opt_state if learner else None
            }
            
            saved_steps = len(data_to_save["current_trajectory"])
            saved_trajs = len(data_to_save["experience_queue"]) + len(data_to_save["batch_buffer"])
            
            try:
                temp_path = traj_checkpoint_path + ".tmp"
                with open(temp_path, "wb") as f:
                    pickle.dump(data_to_save, f)
                os.replace(temp_path, traj_checkpoint_path)
                log(f"/save_trajectory: Saved {saved_steps} steps and {saved_trajs} queued trajectories to {traj_checkpoint_path} (atomic)")
            except Exception as e:
                log(f"/save_trajectory: Error saving trajectory: {e}")
                
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "saved", 
                "saved_steps": saved_steps,
                "saved_queues": saved_trajs
            }).encode())

        elif parsed_path.path == "/load_trajectory":
            # Restore previously saved state from disk
            traj_checkpoint_path = os.path.join(LOG_DIR, "trajectory_checkpoint.pkl")
            loaded_steps = 0
            loaded_trajs = 0
            try:
                if os.path.exists(traj_checkpoint_path):
                    try:
                        with open(traj_checkpoint_path, "rb") as f:
                            data = pickle.load(f)
                    except (EOFError, pickle.UnpicklingError, AttributeError, ImportError) as pe:
                        log(f"/load_trajectory: Corrupted checkpoint file detected! Deleting {traj_checkpoint_path}. Error: {pe}")
                        os.remove(traj_checkpoint_path)
                        data = None

                    if data is not None:
                        if isinstance(data, dict):
                            # Restore current trajectory
                            current_trajectory = data.get("current_trajectory", [])
                            
                            # Restore experience queue
                            q_items = data.get("experience_queue", [])
                            for item in q_items:
                                experience_queue.put(item)
                            
                            # Restore batch buffer
                            b_items = data.get("batch_buffer", [])
                            if training_worker:
                                with training_worker.buffer_lock:
                                    training_worker.batch_buffer.extend(b_items)
                            else:
                                # If worker doesn't exist yet, put them in the queue
                                for item in b_items:
                                    experience_queue.put(item)
                            
                            loaded_steps = len(current_trajectory)
                            loaded_trajs = len(q_items) + len(b_items)
                            
                            # Restore learner state if available
                            if learner:
                                if "learner_params" in data and data["learner_params"] is not None:
                                    log("/load_trajectory: Restoring learner parameters.")
                                    learner.params = data["learner_params"]
                                if "learner_fixed_params" in data and data["learner_fixed_params"] is not None:
                                    log("/load_trajectory: Restoring learner fixed parameters.")
                                    learner.fixed_params = data["learner_fixed_params"]
                                if "learner_opt_state" in data and data["learner_opt_state"] is not None:
                                    log("/load_trajectory: Restoring learner optimizer state.")
                                    learner.opt_state = data["learner_opt_state"]
                            else:
                                log("/load_trajectory: Warning: learner NOT initialized, skipping optimizer state restoration.")
                        else:
                            # Fallback for old list-only format
                            current_trajectory = data
                            loaded_steps = len(current_trajectory)
                            
                        os.remove(traj_checkpoint_path)
                        log(f"/load_trajectory: Restored {loaded_steps} steps and {loaded_trajs} queued trajectories.")
                else:
                    log("/load_trajectory: No trajectory checkpoint file found.")
            except Exception as e:
                log(f"/load_trajectory: Error loading trajectory: {e}")
                traceback.print_exc()

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "loaded", 
                "loaded_steps": loaded_steps,
                "loaded_queues": loaded_trajs
            }).encode())

        elif parsed_path.path == "/start":
            learning_active = True
            log("[Python] Learning started via /start endpoint!")
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
            
        elif parsed_path.path == "/continue_game":
            cmd = "continue_game"
            command_queue.put(cmd)
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "queued", "command": cmd}).encode())
            
        elif parsed_path.path == "/reload_simulator":
            query_components = parse_qs(parsed_path.query)
            v = query_components.get("v", [None])[0]
            
            res = reload_battle_simulator(v)
            
            self.send_response(200 if res == "success" else 500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": res}).encode())
            
        elif parsed_path.path == "/reload_bridge":
            log("[RELOAD] Self-reloading rnad_bridge...")
            try:
                import importlib
                importlib.reload(sys.modules["rnad_bridge"])
                res = "success"
            except Exception as e:
                res = f"Error: {e}"
                log(f"[RELOAD] Bridge reload failed: {res}")
            
            self.send_response(200 if res == "success" else 500)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": res}).encode())
            
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
        # Start server thread immediately
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Start model loading in a background thread to prevent GIL starvation
        # during Godot main thread initialization.
        def deferred_load():
            global initialized
            try:
                load_model()
                initialized = True
                print("[Python] rnad_bridge initialization complete (background).")
            except Exception as e:
                print(f"[Python] Critical error during background model loading: {e}")
                traceback.print_exc()

        loader_thread = threading.Thread(target=deferred_load, daemon=True)
        loader_thread.start()
        
    except Exception as e:
        print(f"[Python] Critical error during initialization: {e}")

if os.environ.get("SKIP_RNAD_INIT") != "1":
    init()
