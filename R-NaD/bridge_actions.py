import json
from bridge_logger import log
from bridge_state import needs_target

def map_action_to_json(action_idx, state, state_type, reward_tracker):
    action = {"action": "wait"}
    
    if action_idx == 99:
        if state_type == "rewards" and not state.get("has_open_potion_slots", True):
            rewards = state.get("rewards", [])
            for i in range(min(len(rewards), 10)):
                if i in reward_tracker.skipped_reward_indices: continue
                reward = rewards[i]
                if "Potion" in reward.get("type", ""):
                    reward_tracker.skipped_reward_indices.add(i)
                    break
        return {"action": "wait"}
        
    elif action_idx == 90:
        return {"action": "confirm_selection"}
        
    elif action_idx == 91:
        return {"action": "open_chest"}
        
    elif 94 <= action_idx <= 98:
        if state_type == "rewards" and not state.get("has_open_potion_slots", True):
            rewards = state.get("rewards", [])
            for i in range(min(len(rewards), 10)):
                if i in reward_tracker.skipped_reward_indices: continue
                reward = rewards[i]
                if "Potion" in reward.get("type", ""):
                    discard_idx = action_idx - 94
                    reward_idx = reward.get("index")
                    reward_tracker.skipped_reward_indices.add(i)
                    return {
                        "action": "discard_and_take_potion",
                        "discard_index": discard_idx,
                        "reward_index": reward_idx
                    }
        return {"action": "discard_potion", "index": action_idx - 94}
    
    elif state_type == "combat":
        hand = state.get("hand") or []
        potions = state.get("potions", [])
        if action_idx < 50:
            card_idx, target_idx = action_idx // 5, action_idx % 5
            if card_idx < len(hand):
                card = hand[card_idx]
                action = {"action": "play_card", "card_id": card.get("id"), "card_index": card_idx}
                if needs_target(card): action["target_index"] = target_idx
        elif 50 <= action_idx < 75:
            potion_linear_idx = action_idx - 50
            potion_idx, target_idx = potion_linear_idx // 5, potion_linear_idx % 5
            if potion_idx < len(potions):
                potion = potions[potion_idx]
                action = {"action": "use_potion", "index": potion_idx}
                target_type = potion.get("targetType", "None")
                if "Enemy" in target_type or "Single" in target_type or "Ally" in target_type or "Player" in target_type:
                    action["target_index"] = target_idx
        elif action_idx == 75: action = {"action": "end_turn"}
        elif action_idx == 86: action = {"action": "proceed"}
    
    elif state_type == "rewards":
        rewards = state.get("rewards", [])
        if 76 <= action_idx < 86:
            reward_idx = action_idx - 76
            if reward_idx < len(rewards):
                action = {"action": "select_reward", "index": reward_idx}
        elif action_idx == 86: action = {"action": "proceed"}
    
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
        elif action_idx == 86: action = {"action": "proceed"}
    
    elif state_type == "shop":
        items = state.get("items", [])
        if action_idx < len(items):
            action = {"action": "buy_item", "index": items[action_idx].get("index")}
        elif action_idx == 86: action = {"action": "shop_proceed"}
    
    elif state_type == "treasure":
        if action_idx == 86: action = {"action": "proceed"}
    
    elif state_type == "treasure_relics":
        relics = state.get("relics", [])
        if action_idx < len(relics):
            action = {"action": "select_treasure_relic", "index": relics[action_idx].get("index")}
    
    elif state_type == "card_reward":
        cards, buttons = state.get("cards", []), state.get("buttons", [])
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
            if state_type == "grid_selection" and state.get("subtype") == "choose_a_card":
                action = {"action": "select_grid_card", "index": -1}
            else:
                action = {"action": "confirm_selection"}
    
    elif state_type == "game_over":
        if action_idx == 86: action = {"action": "proceed"}
        elif action_idx == 87: action = {"action": "return_to_main_menu"}

    return action
