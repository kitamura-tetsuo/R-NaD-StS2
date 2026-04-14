import os
import json
import datetime
import shutil
import filecmp
from bridge_logger import log

# --- Save Data Backup & Restoration Logic ---
SOURCE_A = "/home/ubuntu/.local/share/SlayTheSpire2/steam/76561198725031675/modded/profile1/saves"
SOURCE_B = "/home/ubuntu/.local/share/Steam/userdata/764765947/2868840/remote/modded/profile1/saves"
BACKUP_ROOT = os.path.expanduser("~/sts2_backups")

class BackupManager:
    def __init__(self, backup_root):
        self.backup_root = backup_root
        self.stack = [] # List of {path: str, retry_count: int, reward: float, trials: list, ...}
        self.hp_loss_history = [] # Track HP loss for each trial
        self.current_trial_actions = [] # Actions taken since the last backup or restore
        self.map_blacklist = {} # (floor, current_row, current_col) -> set of target_node_indices that failed
        self.total_retry_count = 0 # Total retries within a single game
        if not os.path.exists(self.backup_root):
            os.makedirs(self.backup_root, exist_ok=True)
            
    def clear(self):
        """Clear all backups in the stack."""
        log(f"[BackupManager] Clearing stack of size {len(self.stack)}")
        self.stack = []
        self.hp_loss_history = []
        self.current_trial_actions = []
        self.map_blacklist = {}
        self.total_retry_count = 0 

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

    def backup(self, current_reward, is_elite=False, is_boss=False, floor=0, is_map_branch=False, next_nodes=None, current_row=-1, current_col=-1):
        """Create a backup if the state has changed."""
        if self.stack:
            last_backup = self.stack[-1]["path"]
            if (self._are_saves_identical(last_backup)):
                log(f"[BackupManager] Saves identical to latest backup. Checking combat status.")
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
                "accumulated_reward": float(current_reward),
                "is_elite": is_elite,
                "is_boss": is_boss,
                "floor": int(floor),
                "is_map_branch": is_map_branch,
                "next_nodes": next_nodes,
                "current_row": current_row,
                "current_col": current_col
            }
            with open(os.path.join(backup_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)
                
            self.stack.append({
                "path": backup_dir,
                "retry_count": 0,
                "reward": current_reward,
                "trials": [],
                "is_elite": is_elite,
                "is_boss": is_boss,
                "floor": int(floor),
                "win_hp_losses": [],
                "is_map_branch": is_map_branch,
                "next_nodes": next_nodes,
                "current_row": current_row,
                "current_col": current_col
            })
            log(f"[BackupManager] Created new backup at {backup_dir} (Elite={is_elite}, Boss={is_boss}, Floor={floor}). Stack size: {len(self.stack)}")
            self.current_trial_actions = []
            self.hp_loss_history = []
            is_combat = self._is_combat_save(backup_dir)
            if not is_combat:
                log(f"[BackupManager] Warning: Created backup is not a combat save. Will retry next turn.")
            return is_combat
        except Exception as e:
            log(f"[BackupManager] ERROR during backup: {e}")
            return False

    def _is_combat_save(self, backup_dir):
        """Check if the backup contains an active combat state."""
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
                    return True
                return not pfr.get("is_pre_finished", True)
        except Exception as e:
            log(f"[BackupManager] _is_combat_save: Error checking {save_path}: {e}")
            return False

    def record_hp_loss(self, hp_loss, is_victory=False):
        """Record HP loss for the current trial."""
        if self.stack:
            latest = self.stack[-1]
            if is_victory:
                latest["win_hp_losses"].append(hp_loss)
            
        self.hp_loss_history.append(hp_loss)
        log(f"[BackupManager] Recorded {'Victory' if is_victory else 'Defeat'} HP Loss: {hp_loss}.")

    def check_hp_performance(self, hp_loss):
        """Check if the current HP loss meets the criteria to proceed to the next floor."""
        if not self.stack:
            return True, 0, 3

        latest = self.stack[-1]
        is_elite = latest.get("is_elite", False)
        is_boss = latest.get("is_boss", False)
        floor = latest.get("floor", 0)
        retry_count = latest["retry_count"]
        win_hp_losses = latest.get("win_hp_losses", [])
        
        max_retries = 3
        if is_elite or is_boss:
            if retry_count >= 10 and win_hp_losses:
                max_retries = 20
            else:
                max_retries = 10

        if retry_count < 10 or not (is_elite or is_boss):
            should_proceed = hp_loss <= floor
            goal_desc = f"HP loss <= {floor} (Stage 1)"
        else:
            if not win_hp_losses:
                should_proceed = hp_loss <= floor
                goal_desc = f"HP loss <= {floor} (Stage 2 Failback)"
            else:
                sorted_wins = sorted(win_hp_losses)
                n = len(sorted_wins)
                median_idx = (n - 1) // 2
                median_val = sorted_wins[median_idx]
                should_proceed = hp_loss <= median_val
                goal_desc = f"HP loss <= median({median_val}) (Stage 2)"

        log(f"[BackupManager] Performance Check: loss={hp_loss}, {goal_desc}, should_proceed={should_proceed}, retries={retry_count}/{max_retries}")
        return should_proceed, retry_count, max_retries

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
        """Return a set of actions that have been taken in past trials."""
        if not self.stack:
            return set()
        
        latest = self.stack[-1]
        trials = latest.get("trials", [])
        if not trials:
            return set()
        
        current_len = len(self.current_trial_actions)
        penalized = set()
        for trial in trials:
            if len(trial) > current_len and trial[:current_len] == self.current_trial_actions:
                penalized.add(trial[current_len])
        
        return penalized

    def restore(self, force=False):
        """Restore the latest valid backup, backtracking if necessary."""
        while self.stack:
            latest = self.stack[-1]
            
            is_elite = latest.get("is_elite", False)
            is_boss = latest.get("is_boss", False)
            win_hp_losses = latest.get("win_hp_losses", [])
            retry_count = latest["retry_count"]
            
            current_max = 3
            if is_elite or is_boss:
                if retry_count >= 10 and win_hp_losses:
                    current_max = 20
                else:
                    current_max = 10

            log(f"[BackupManager] Restore attempt for floor {latest.get('floor')} (Retry {retry_count}/{current_max}). Force={force}")
            if retry_count < current_max or force:
                retry_msg = f"Retry {retry_count+1}/{current_max}" if not force else "FORCED Retry"
                log(f"[BackupManager] Restoring {os.path.basename(latest['path'])} ({retry_msg})")
                
                try:
                    if latest.get("is_map_branch") and self.current_trial_actions:
                        failed_route_idx = self.current_trial_actions[0]
                        if failed_route_idx < 50: 
                            floor = latest.get("floor")
                            row = latest.get("current_row")
                            col = latest.get("current_col")
                            key = (floor, row, col)
                            if key not in self.map_blacklist:
                                self.map_blacklist[key] = set()
                            
                            if failed_route_idx not in self.map_blacklist[key]:
                                log(f"[BackupManager] Blacklisting failed route index {failed_route_idx} for map node at floor {floor}, pos ({row}, {col}).")
                                self.map_blacklist[key].add(failed_route_idx)
                            
                            next_nodes = latest.get("next_nodes", [])
                            if len(self.map_blacklist[key]) >= len(next_nodes):
                                log(f"[BackupManager] ALL routes exhausted at floor {floor}, pos ({row}, {col}). Recursive backtracking...")
                                self.stack.pop()
                                continue 
                    
                    appdata_backup = os.path.join(latest["path"], "AppData")
                    userdata_backup = os.path.join(latest["path"], "UserData")
                    
                    if os.path.exists(appdata_backup):
                        shutil.copytree(appdata_backup, SOURCE_A, dirs_exist_ok=True)
                    if os.path.exists(userdata_backup):
                        shutil.copytree(userdata_backup, SOURCE_B, dirs_exist_ok=True)
                    
                    latest["retry_count"] += 1
                    self.total_retry_count += 1 
                    return latest["reward"]
                except Exception as e:
                    log(f"[BackupManager] ERROR during restore: {e}")
                    return None
            else:
                log(f"[BackupManager] !!! BACKTRACKING !!! Backup {os.path.basename(latest['path'])} exhausted. Popping...")
                self.stack.pop()
        
        log(f"[BackupManager] FAILED: All backups exhausted or stack empty.")
        return None
