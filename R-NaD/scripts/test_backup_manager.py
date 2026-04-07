
import os
import json
import shutil
from datetime import datetime

# Mock log function
def log(msg):
    print(f"DEBUG: {msg}")

# Mock globals for BackupManager
SOURCE_A = "/home/ubuntu/src/R-NaD-StS2/R-NaD/mock_appdata"
SOURCE_B = "/home/ubuntu/src/R-NaD-StS2/R-NaD/mock_userdata"

class BackupManager:
    def __init__(self, backup_root):
        self.backup_root = backup_root
        self.stack = []
        self.hp_loss_history = []
        self.current_trial_actions = []
        if not os.path.exists(self.backup_root):
            os.makedirs(self.backup_root, exist_ok=True)

    def _is_combat_save(self, path): return True # Mock
    def _are_saves_identical(self, path): return False # Mock

    def backup(self, current_reward, is_elite=False, is_boss=False, floor=0):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = os.path.join(self.backup_root, f"backup_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)
        
        self.stack.append({
            "path": backup_dir,
            "retry_count": 0,
            "reward": current_reward,
            "trials": [],
            "is_elite": is_elite,
            "is_boss": is_boss,
            "floor": int(floor),
            "win_hp_losses": []
        })
        log(f"Created backup (Elite={is_elite}, Boss={is_boss}, Floor={floor})")
        return True

    def record_hp_loss(self, hp_loss, is_victory=False):
        if self.stack:
            latest = self.stack[-1]
            if is_victory:
                latest["win_hp_losses"].append(hp_loss)
        log(f"Recorded {'Victory' if is_victory else 'Defeat'} HP Loss: {hp_loss}")

    def check_hp_performance(self, hp_loss):
        if not self.stack: return True, 0, 3
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

        log(f"Performance Check: loss={hp_loss}, {goal_desc}, should_proceed={should_proceed}, retries={retry_count}/{max_retries}")
        return should_proceed, retry_count, max_retries

    def restore(self, force=False):
        if not self.stack: return None
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

        if retry_count < current_max or force:
            latest["retry_count"] += 1
            log(f"Restored (Retry {latest['retry_count']}/{current_max})")
            return latest["reward"]
        else:
            log(f"Exhausted at {retry_count}/{current_max}. Backtracking.")
            self.stack.pop()
            return self.restore()

# Test Scenarios
def run_tests():
    root = "/home/ubuntu/src/R-NaD-StS2/R-NaD/scripts/test_backups"
    if os.path.exists(root): shutil.rmtree(root)
    bm = BackupManager(root)

    print("\n--- Scenario 1: Normal Enemy (Floor 10) ---")
    bm.backup(100, is_elite=False, is_boss=False, floor=10)
    # 1. Defeat
    bm.record_hp_loss(50, is_victory=False)
    bm.restore() # retry_count: 1
    # 2. Victory but high HP loss
    bm.record_hp_loss(15, is_victory=True)
    res, retries, limit = bm.check_hp_performance(15) # Should be False (15 > 10)
    if not res: bm.restore() # retry_count: 2
    # 3. Victory and low HP loss
    bm.record_hp_loss(5, is_victory=True)
    res, retries, limit = bm.check_hp_performance(5) # Should be True (5 <= 10)
    print(f"Result: Proceed={res}, Final Retries={retries}")

    print("\n--- Scenario 2: Elite (Floor 10) - Stage 1 Success ---")
    bm.stack = []
    bm.backup(200, is_elite=True, floor=10)
    bm.record_hp_loss(8, is_victory=True)
    res, retries, limit = bm.check_hp_performance(8) # Should be True (8 <= 10)
    print(f"Result: Proceed={res}")

    print("\n--- Scenario 3: Elite (Floor 10) - Stage 2 Transition ---")
    bm.stack = []
    bm.backup(300, is_elite=True, floor=10)
    for i in range(10):
        bm.record_hp_loss(20 + i, is_victory=True) # All wins > 10
        res, retries, limit = bm.check_hp_performance(20 + i)
        print(f"Try {i+1}: res={res}, retries={retries}/{limit}")
        bm.restore()
    
    # 11th try (retry_count = 10). Should use Median. 
    # Wins so far: 20, 21, 22, 23, 24, 25, 26, 27, 28, 29. Median is 24.
    bm.record_hp_loss(24, is_victory=True)
    res, retries, limit = bm.check_hp_performance(24) # Should be True (24 <= 24)
    print(f"Result: Proceed={res}, Retries={retries}/{limit}")

    print("\n--- Scenario 4: Defeat Exhaustion ---")
    bm.stack = []
    bm.backup(400, is_elite=False, floor=10) # Normal enemy
    for i in range(3):
        bm.record_hp_loss(50, is_victory=False)
        r = bm.restore()
        print(f"Restore {i+1} result: {r}")
    
    print("4th defeat...")
    bm.record_hp_loss(50, is_victory=False)
    r = bm.restore() # Should return None/Backtrack
    print(f"Restore 4 result (should be backtrack/None): {r}")

if __name__ == "__main__":
    run_tests()
