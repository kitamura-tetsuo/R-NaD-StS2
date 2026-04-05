import json
import os
import time
import asyncio
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from watchfiles import awatch

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LIVE_STATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "tmp/live_state.json"))

# Event History Persistence
event_history = []
prev_state_type = None
prev_action_idx = None
turn_counter = 1
history_lock = asyncio.Lock()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

def get_action_name(idx: int, state: Dict) -> str:
    """Map action index to a human-readable name using the current state."""
    state_type = state.get("type", "unknown")
    
    if idx == 75:
        return "End Turn"
    if idx == 86:
        return "Proceed"
    if idx == 87:
        return "Return to Main Menu"
    if idx == 90:
        return "Confirm / Skip"
    if idx == 91:
        return "Open Chest"
    if idx == 99:
        return "Wait"
        
    if state_type == "combat":
        if 0 <= idx < 50:
            card_idx = idx // 5
            target_idx = idx % 5
            hand = state.get("hand", [])
            enemies = state.get("enemies", [])
            
            card_name = hand[card_idx].get("name", f"Card {card_idx}") if card_idx < len(hand) else f"Card {card_idx}"
            
            if target_idx < len(enemies):
                target_name = enemies[target_idx].get("name", f"Enemy {target_idx}")
                return f"Play {card_name} on {target_name}"
            return f"Play {card_name}"
            
        if 50 <= idx < 75:
            pot_idx = (idx - 50) // 5
            target_idx = (idx - 50) % 5
            potions = state.get("potions", [])
            enemies = state.get("enemies", [])
            
            pot_name = potions[pot_idx].get("name", f"Potion {pot_idx}") if pot_idx < len(potions) else f"Potion {pot_idx}"
            if target_idx < len(enemies):
                target_name = enemies[target_idx].get("name", f"Enemy {target_idx}")
                return f"Use {pot_name} on {target_name}"
            return f"Use {pot_name}"

    if state_type == "rewards" and 76 <= idx < 86:
        reward_idx = idx - 76
        rewards = state.get("rewards", [])
        if reward_idx < len(rewards):
            return f"Pick {rewards[reward_idx].get('description', f'Reward {reward_idx}')}"
        return f"Pick Reward {reward_idx}"

    if state_type == "map" and 88 <= idx < 90:
        return f"Move to Room {idx - 88}"

    return f"Action {idx}"

async def process_state(data: Dict) -> Dict:
    """Enhance raw state with human-readable action names and detect events."""
    global prev_state_type, prev_action_idx, turn_counter, event_history
    
    state = data.get("state", {})
    probs = data.get("probs", [])
    action_idx = data.get("action_idx")
    timestamp = data.get("timestamp", time.time())
    
    state_type = state.get("type", "unknown")
    
    # Event Detection
    event_label = ""
    event_color = "#666"
    
    async with history_lock:
        if data.get("reset"):
            event_history = []
            prev_state_type = None
            prev_action_idx = None
            turn_counter = 1
            
        if prev_state_type and state_type != prev_state_type:
            if state_type == 'combat':
                event_label = "COMBAT START"
                event_color = "#ef4444"
                turn_counter = 1
            elif state_type == 'rewards' and prev_state_type != 'rewards':
                event_label = "VICTORY / REWARDS" if prev_state_type == 'combat' else "REWARDS"
                event_color = "#22c55e"
            elif state_type == 'map':
                event_label = "MAP"
                event_color = "#3b82f6"
            elif state_type == 'event':
                event_label = "EVENT"
                event_color = "#a855f7"
            elif state_type == 'game_over':
                event_label = "GAME OVER"
                event_color = "#f87171" # Light red
        elif state_type == 'combat':
            if prev_action_idx == 75:
                turn_counter += 1
                event_label = f"TURN {turn_counter}"
                event_color = "#eab308"
            elif prev_state_type != 'combat':
                event_label = "TURN 1"
                event_color = "#eab308"
                turn_counter = 1
        
        if event_label:
            event_history.append({"time": timestamp, "label": event_label, "color": event_color})
            if len(event_history) > 40:
                event_history.pop(0)

        prev_state_type = state_type
        prev_action_idx = action_idx
        data["events"] = list(event_history)
    
    if probs:
        import numpy as np
        probs_arr = np.array(probs)
        
        # Get executed action info
        selected_prob = float(probs_arr[action_idx]) if action_idx < len(probs_arr) else 0.0
        selected_action = {
            "id": int(action_idx),
            "name": get_action_name(int(action_idx), state),
            "prob": selected_prob,
            "isSelected": True,
            "isSearch": bool(data.get("is_search", False))
        }
        
        # Get top 3 others (excluding selected)
        # Create mask to hide selected action
        probs_others = probs_arr.copy()
        if action_idx < len(probs_others):
            probs_others[action_idx] = -1.0
            
        top_other_indices = np.argsort(probs_others)[-3:][::-1]
        top_actions = [selected_action]
        
        for i in top_other_indices:
            if i < len(probs_others) and probs_others[i] > 0.0001: # Small threshold
                top_actions.append({
                    "id": int(i),
                    "name": get_action_name(int(i), state),
                    "prob": float(probs_others[i]),
                    "isSelected": False,
                    "isSearch": False
                })
                
        data["top_actions"] = top_actions
        
    return data

async def watch_file():
    """Watch the live_state.json file and broadcast updates."""
    last_mtime = 0
    while True:
        try:
            if os.path.exists(LIVE_STATE_PATH):
                mtime = os.path.getmtime(LIVE_STATE_PATH)
                if mtime > last_mtime:
                    last_mtime = mtime
                    with open(LIVE_STATE_PATH, "r") as f:
                        raw_data = json.load(f)
                        processed_data = await process_state(raw_data)
                        await manager.broadcast(json.dumps(processed_data))
            await asyncio.sleep(0.1) # Check every 100ms
        except Exception as e:
            print(f"Error watching file: {e}")
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(watch_file())

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    # Send initial state if available
    try:
        if os.path.exists(LIVE_STATE_PATH):
            with open(LIVE_STATE_PATH, "r") as f:
                raw_data = json.load(f)
                processed_data = await process_state(raw_data)
                await websocket.send_text(json.dumps(processed_data))
    except Exception:
        pass
        
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8051)
