import os
import json
import mmap
import struct
import glob
import sys
import importlib.util
import traceback
import numpy as np
from bridge_logger import log
from bridge_state import needs_target
from bridge_vocab import CARD_VOCAB, MONSTER_VOCAB, POWER_VOCAB, BOSS_VOCAB, POTION_VOCAB

DISCREPANCY_LOG_DIR = "/home/ubuntu/src/R-NaD-StS2/battle_simulator/discrepancy_logs"
VALIDATION_SKIP_RANDOM = {"SWORD_BOOMERANG", "FIEND_FIRE", "HAVOC", "INFERNAL_BLADE"}
VALIDATION_SKIP_DRAW = {"POMMEL_STRIKE", "SHRUG_IT_OFF", "WARCRY", "OFFERING", "BATTLE_TRANCE", 
                        "BURNING_PACT", "HAVOC", "DARK_EMBRACE", "EVOLVE"}

# Global simulator instance
battle_simulator = None

class SimulatorManager:
    def __init__(self, bridge_dir):
        self.bridge_dir = bridge_dir
        self.shm_path = os.path.join(self.bridge_dir, "tmp/sts2_sim_shm")
        self.shm_size = 10 * 1024 * 1024 # 10MB
        self.shm = None
        # Must match Rust encoder: GLOBAL_SIZE + COMBAT_SIZE + BOW_SIZE * 4 + 2
        self.tensor_size = 512 + 384 + 611 * 4 + 2
        
    def init_simulator(self, sim):
        sim.set_vocabulary(CARD_VOCAB, MONSTER_VOCAB, POWER_VOCAB, BOSS_VOCAB, POTION_VOCAB)
        os.makedirs(os.path.dirname(self.shm_path), exist_ok=True)
        sim.init_shm(self.shm_path, self.shm_size)
        if os.path.exists(self.shm_path):
            f = open(self.shm_path, "r+b")
            self.shm = mmap.mmap(f.fileno(), self.shm_size)
            log(f"[SimulatorManager] Shared memory mapped at {self.shm_path}")

    def read_results(self):
        if not self.shm: return []
        try:
            self.shm.seek(0)
            num_results_data = self.shm.read(4)
            if not num_results_data or len(num_results_data) < 4: return []
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
                results.append({"actions": actions, "outcomes": outcomes})
            return results
        except Exception as e:
            log(f"[SimulatorManager] Error reading SHM: {e}")
            return []

class CombatValidator:
    def __init__(self):
        self.last_state = None
        self.last_action_idx = None
        self.enabled = True

    def to_simulator_json(self, cs_state):
        player_data = cs_state.get("player", {}) or {}
        def convert_powers(powers):
            return [{"id": p.get("id", ""), "amount": p.get("amount", 0)} for p in (powers or [])]
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
            if isinstance(c, str): c = {"id": c}
            tt_map = {"AnyEnemy": "Single", "SingleEnemy": "Single", "AllEnemies": "All", "AllEnemy": "All", "Self": "SelfTarget", "None": "None"}
            return {
                "id": c.get("id", "Unknown"), "cost": c.get("cost", 0), "base_damage": c.get("baseDamage", 0),
                "base_block": c.get("baseBlock", 0), "currentDamage": c.get("currentDamage", 0), "currentBlock": c.get("currentBlock", 0),
                "magic_number": c.get("magicNumber", 0), "target": tt_map.get(c.get("targetType", "None"), "None"),
                "is_upgraded": c.get("upgraded", False), "isPlayable": c.get("isPlayable", True)
            }
        sim_state = {
            "player": convert_creature(player_data, is_player=True),
            "enemies": [convert_creature(e) for e in cs_state.get("enemies", [])],
            "hand": [convert_card(c) for c in cs_state.get("hand", [])],
            "draw_pile": [convert_card(c) for c in cs_state.get("drawPile", [])],
            "discard_pile": [convert_card(c) for c in cs_state.get("discardPile", [])],
            "exhaust_pile": [convert_card(c) for c in cs_state.get("exhaustPile", [])],
            "potions": [{"id": p.get("id", "empty"), "name": p.get("name", "Empty Slot"), "can_use": p.get("canUse", False), "targetType": p.get("targetType", "None")} for p in cs_state.get("potions", [])],
            "energy": player_data.get("energy", 0), "max_energy": player_data.get("maxEnergy", 0), "stars": player_data.get("stars", 0), "retains_block": cs_state.get("retains_block", False), "floor": cs_state.get("floor", 1)
        }
        return json.dumps(sim_state)

    def validate(self, current_state_json, bridge_globals):
        if not self.enabled or self.last_state is None or self.last_action_idx is None: return
        last_state, last_action_idx = self.last_state, self.last_action_idx
        self.last_state, self.last_action_idx = None, None
        try:
            current_state = json.loads(current_state_json)
            if current_state.get("type") != "combat": return
            sim_json = self.to_simulator_json(last_state)
            global battle_simulator
            if battle_simulator is None: return
            sim = battle_simulator.Simulator.from_json(sim_json)
            if last_action_idx < 50:
                card_idx, target_idx = last_action_idx // 5, last_action_idx % 5
                hand = last_state.get("hand", [])
                if card_idx < len(hand):
                    card = hand[card_idx]
                    t_val = target_idx if needs_target(card) else None
                    sim.play_card(card_idx, t_val)
            else: return
            sim_outcome = json.loads(sim.get_state_json())
            discrepancies = []
            p_real, p_sim = current_state.get("player", {}), sim_outcome["player"]
            if p_real.get("hp") != p_sim["cur_hp"]: discrepancies.append(f"Player HP: real={p_real.get('hp')}, sim={p_sim['cur_hp']}")
            if p_real.get("block") != p_sim["block"]: discrepancies.append(f"Player Block: real={p_real.get('block')}, sim={p_sim['block']}")
            if p_real.get("energy") != sim_outcome.get("energy"): discrepancies.append(f"Player Energy: real={p_real.get('energy')}, sim={sim_outcome.get('energy')}")
            e_real, e_sim = current_state.get("enemies", []), sim_outcome["enemies"]
            for i in range(min(len(e_real), len(e_sim))):
                if e_real[i].get("hp") != e_sim[i]["cur_hp"]: discrepancies.append(f"Enemy {i} HP: real={e_real[i].get('hp')}, sim={e_sim[i]['cur_hp']}")
                if e_real[i].get("block") != e_sim[i]["block"]: discrepancies.append(f"Enemy {i} Block: real={e_real[i].get('block')}, sim={e_sim[i]['block']}")
            if discrepancies:
                card_id = "UNKNOWN"
                if last_action_idx < 50:
                    card_idx = last_action_idx // 5
                    hand = last_state.get("hand", [])
                    if card_idx < len(hand): card_id = hand[card_idx].get("id", "UNKNOWN").upper().split('+')[0].strip()
                is_random_target = card_id in VALIDATION_SKIP_RANDOM and len(last_state.get("enemies", [])) > 1
                is_draw = card_id in VALIDATION_SKIP_DRAW
                filtered = [d for d in discrepancies if not (is_random_target and "Enemy" in d) and not (is_draw and ("HP" in d or "Block" in d))]
                if not filtered:
                    log(f"[SIMULATOR VALIDATION] SUCCESS (Filtered volatile elements for {card_id})")
                    return
                log(f"[SIMULATOR VALIDATION] DISCREPANCY FOUND after action {last_action_idx} ({card_id}):")
                for d in filtered: log(f"  - {d}")
                log(f"  Sim State: {json.dumps(sim_outcome)}")
                # Screenshot and logging (simplified for modularity)
                self._log_discrepancy(last_state, current_state, last_action_idx, card_id, discrepancies, filtered, bridge_globals)
            else: log(f"[SIMULATOR VALIDATION] SUCCESS after action {last_action_idx}")
        except Exception as e: log(f"[SIMULATOR VALIDATION] ERROR: {e}")

    def _log_discrepancy(self, last_state, current_state, last_action_idx, card_id, discrepancies, filtered, bridge_globals):
        try:
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            os.makedirs(DISCREPANCY_LOG_DIR, exist_ok=True)
            with open(os.path.join(DISCREPANCY_LOG_DIR, f"state_before_{ts}.json"), "w") as f: json.dump(last_state, f, indent=2)
            with open(os.path.join(DISCREPANCY_LOG_DIR, f"state_after_{ts}.json"), "w") as f: json.dump(current_state, f, indent=2)
            action_info = {"action_idx": last_action_idx, "discrepancies": discrepancies, "card_id": card_id}
            with open(os.path.join(DISCREPANCY_LOG_DIR, f"action_{ts}.json"), "w") as f: json.dump(action_info, f, indent=2)
            # Signal screenshot
            screenshot_path = os.path.join(DISCREPANCY_LOG_DIR, f"screenshot_{ts}.png")
            bridge_globals['request_screenshot'](screenshot_path)
            log(f"  [DISCREPANCY] Requesting screenshot: {screenshot_path}")
        except Exception as e: log(f"  [DISCREPANCY LOG ERROR] {e}")

def reload_battle_simulator(bridge_dir, current_validator=None):
    global battle_simulator
    try:
        sos = glob.glob(os.path.join(bridge_dir, "battle_simulator_*.so"))
        if not sos: target_so = os.path.join(bridge_dir, "battle_simulator.so")
        else:
            sos.sort(key=os.path.getmtime)
            target_so = sos[-1]
        if not os.path.exists(target_so): return f"Error: {target_so} not found"
        log(f"[RELOAD] Loading simulator from: {target_so}")
        spec = importlib.util.spec_from_file_location("battle_simulator", target_so)
        if spec is None: return "Error: Could not create spec"
        new_mod = importlib.util.module_from_spec(spec)
        sys.modules["battle_simulator"] = new_mod
        spec.loader.exec_module(new_mod)
        battle_simulator = new_mod
        log(f"[RELOAD] Successfully reloaded battle_simulator")
        return "success"
    except Exception as e:
        err = f"Error during reload: {e}"
        log(f"[RELOAD] {err}")
        return err
