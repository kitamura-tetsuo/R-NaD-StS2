from bridge_logger import log

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
        self.last_enemy_count: int = -1
        self.last_predicted_damage_to_player: float = 0.0
        self.last_action_idx: int = -1
        self.was_elite: bool = False
        self.was_boss: bool = False

    def reset_for_new_run(self, backup_manager=None):
        """Reset state when returning to main menu or starting a fresh run."""
        if self.last_state_type == "main_menu":
            return # Avoid redundant resets and logging noise

        self.last_processed_floor = -1
        self.last_player_hp = 0
        self.last_total_enemy_hp = 0
        self.last_reward_floor = -1
        self.last_selected_reward_idx = None
        self.skipped_reward_indices = set()
        self.session_cumulative_reward = 0.0
        self.episode_end_recorded = False
        self.combat_initialized = False
        self.last_state_type = "main_menu" # Set immediately to prevent re-entry noise
        self.last_upgraded_count = -1
        self.last_total_cards = -1
        self.last_potion_count = -1
        self.last_enemy_count = -1
        self.last_predicted_damage_to_player = 0.0
        self.last_action_idx = -1
        self.was_elite = False
        self.was_boss = False
        if backup_manager:
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

    def initialize_combat(self, hp, enemy_hp, enemy_count, predicted_damage_to_player):
        """Initialize combat trackers to avoid the start-of-combat penalty."""
        self.last_player_hp = hp
        self.last_total_enemy_hp = enemy_hp
        self.last_enemy_count = enemy_count
        self.last_predicted_damage_to_player = predicted_damage_to_player
        self.last_action_idx = -1
        self.combat_initialized = True
        log(f"RewardTracker: Combat initialized (Player HP: {hp}, Enemy HP: {enemy_hp}, Count: {enemy_count}, PredDamage: {predicted_damage_to_player})")

def compute_reward(state, reward_tracker, state_type=None):
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

def compute_intermediate_reward(state, reward_tracker, state_type, action_idx):
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
        # Update trackers when floor changes
        reward_tracker.last_processed_floor = current_floor
        reward_tracker.last_player_hp = current_hp
        reward_tracker.last_total_enemy_hp = current_enemy_hp
        reward_tracker.combat_initialized = False # Reset combat init on floor change

    # Combat delta reward (Dense Reward)
    if state_type == "combat":
        # Predicted Damage Delta Reward
        current_pred_damage = float(state.get("predicted_total_damage", 0))
        current_pred_block = float(state.get("predicted_end_block", 0))
        current_pred_dmg_to_player = max(0.0, current_pred_damage - current_pred_block)

        current_enemy_count = sum(1 for e in enemies if e.get("hp", 0) > 0 and not e.get("isMinion", False))
        if not reward_tracker.combat_initialized:
            reward_tracker.initialize_combat(current_hp, current_enemy_hp, current_enemy_count, current_pred_dmg_to_player)
        
        last_hp = reward_tracker.last_player_hp
        last_enemy_hp = reward_tracker.last_total_enemy_hp
        
        # Combat delta reward: Reward damage dealt and HP changes
        damage_dealt = max(0.0, float(last_enemy_hp - current_enemy_hp))
        hp_delta = float(current_hp - last_hp)
        
        combat_reward = (damage_dealt * 0.001) + (hp_delta * 0.03)
        
        if abs(combat_reward) > 1e-6:
            intermediate_reward += combat_reward

        # Reward for reduction of predicted damage (excluding End Turn)
        if reward_tracker.last_action_idx != 75:
            damage_reduction = reward_tracker.last_predicted_damage_to_player - current_pred_dmg_to_player
            if damage_reduction > 0:
                reduction_reward = damage_reduction * 0.004
                intermediate_reward += reduction_reward
        
        # Enemy Defeat Reward: 0.01 per enemy defeated
        last_enemy_count = reward_tracker.last_enemy_count
        if last_enemy_count != -1 and current_enemy_count < last_enemy_count:
            defeat_count = last_enemy_count - current_enemy_count
            defeat_reward = defeat_count * 0.01
            intermediate_reward += defeat_reward

        # Track if it was an elite or boss room
        room_type = state.get("room_type", "")
        if room_type == "Elite":
            reward_tracker.was_elite = True
        elif room_type == "Boss":
            reward_tracker.was_boss = True

        # Update trackers for next step
        reward_tracker.last_player_hp = current_hp
        reward_tracker.last_total_enemy_hp = current_enemy_hp
        reward_tracker.last_enemy_count = current_enemy_count
        reward_tracker.last_predicted_damage_to_player = current_pred_dmg_to_player
        reward_tracker.last_action_idx = action_idx

    # Battle Clear Reward
    if reward_tracker.last_state_type == "combat" and (state_type == "rewards" or state_type == "map"):
        intermediate_reward += 0.1 # Battle clear base
        if reward_tracker.was_elite:
            intermediate_reward += 0.1
        if reward_tracker.was_boss:
            intermediate_reward += 0.5
        reward_tracker.was_elite = False
        reward_tracker.was_boss = False

    # Card / Potion Acquisition Reward (+0.01)
    potions = state.get("potions", [])
    current_potion_count = sum(1 for p in potions if p.get("id") != "empty")
    if reward_tracker.last_potion_count != -1:
        if current_potion_count > reward_tracker.last_potion_count:
            if state_type in ["rewards", "shop", "treasure"]: # After battle reward, shop buy, or treasure
                intermediate_reward += 0.01
        elif current_potion_count < reward_tracker.last_potion_count:
            # 0.001 penalty for using/losing a potion to prevent wasteful use
            intermediate_reward -= 0.001
    reward_tracker.last_potion_count = current_potion_count

    # Card Acquisition Count
    total_cards = -1
    if state_type == "combat":
        # Sum piles as proxy for deck size
        total_cards = (len(state.get("drawPile", [])) + 
                       len(state.get("discardPile", [])) + 
                       len(state.get("exhaustPile", [])) + 
                       len(state.get("hand", [])))
    elif state_type == "grid_selection" or state_type == "card_reward":
        total_cards = len(state.get("cards", []))

    if reward_tracker.last_total_cards != -1 and total_cards > reward_tracker.last_total_cards:
        if state_type != "combat" or reward_tracker.last_state_type == "combat": # Transition or outside
            intermediate_reward += 0.01
    reward_tracker.last_total_cards = total_cards

    # Card Upgrade Reward (+0.01)
    if state_type == "grid_selection" and state.get("subtype") == "NDeckUpgradeSelectScreen":
        cards = state.get("cards", [])
        upgraded_count = sum(1 for c in cards if c.get("upgraded"))
        if reward_tracker.last_upgraded_count != -1 and upgraded_count > reward_tracker.last_upgraded_count:
            intermediate_reward += 0.01
        reward_tracker.last_upgraded_count = upgraded_count
    else:
        reward_tracker.last_upgraded_count = -1

    reward_tracker.last_state_type = state_type
        
    return intermediate_reward
