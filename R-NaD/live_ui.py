import streamlit as st
import json
import os
import pandas as pd
import time
import glob
import numpy as np
from datetime import datetime

# Configuration
LIVE_STATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "tmp/live_state.json"))
TRAJECTORY_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "trajectories"))

st.set_page_config(page_title="StS2 AI Live Monitor", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
    .card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding: 10px;
    }
    .card-box {
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #444;
        font-family: 'monospace';
        font-size: 0.9em;
        background-color: #262730;
        color: #efefef;
    }
    .card-generated {
        border: 1px solid #00ffcc;
        box-shadow: 0 0 5px #00ffcc;
        background-color: #1a2e2a;
    }
    .card-playable {
        color: #00ff00;
    }
    .card-unplayable {
        color: #ff4b4b;
    }
    .section-header {
        color: #ffaa00;
        font-weight: bold;
        margin-top: 20px;
        border-bottom: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

def load_live_state():
    if not os.path.exists(LIVE_STATE_PATH):
        return None
    try:
        with open(LIVE_STATE_PATH, "r") as f:
            return json.load(f)
    except:
        return None

def get_current_trajectory_history():
    # Find the most recent trajectory file
    files = sorted(glob.glob(os.path.join(TRAJECTORY_DIR, "traj_*.json")), reverse=True)
    if not files:
        return []
    
    try:
        with open(files[0], "r") as f:
            data = json.load(f)
            return data.get("steps", [])
    except:
        return []

def render_cards(cards, title, highlight_generated=True):
    st.markdown(f"<div class='section-header'>{title} ({len(cards)})</div>", unsafe_allow_html=True)
    if not cards:
        st.write("None")
        return

    # If it's a list of strings, convert to basic objects
    processed_cards = []
    for c in cards:
        if isinstance(c, str):
            processed_cards.append({"name": c, "id": c, "is_generated": False})
        else:
            processed_cards.append(c)

    cols = st.columns(min(len(processed_cards), 8) or 1)
    for i, card in enumerate(processed_cards):
        with cols[i % 8]:
            is_gen = card.get("is_generated", False)
            is_playable = card.get("isPlayable", True)
            
            gen_class = "card-generated" if is_gen and highlight_generated else ""
            play_class = "card-playable" if is_playable else "card-unplayable"
            
            card_html = f"""
                <div class="card-box {gen_class}">
                    <span class="{play_class}">{card.get('name', '???')}</span><br/>
                    <small style="opacity: 0.6;">{card.get('cost', '')}E {card.get('id', '')}</small>
                </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

def main():
    st.title("🛡️ StS2 R-NaD Inference Monitor")
    
    live_data = load_live_state()
    if not live_data:
        st.info("Waiting for live state data... Ensure the game and bridge are running.")
        time.sleep(2)
        st.rerun()
        return

    state = live_data.get("state", {})
    state_type = state.get("type", "unknown")
    
    # Header Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Floor", state.get("floor", 0))
    with col2:
        hp = state.get("player", {}).get("hp", 0)
        max_hp = state.get("player", {}).get("maxHp", 0)
        st.metric("Player HP", f"{hp}/{max_hp}")
    with col3:
        st.metric("Energy", state.get("player", {}).get("energy", 0))
    with col4:
        v_val = live_data.get("predicted_v", 0.0)
        st.metric("Predicted V", f"{v_val:.4f}")
    with col5:
        st.metric("State", state_type.upper())

    # Main Content Layout
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # Combat View
        if state_type == "combat":
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                st.markdown("<div class='section-header'>Enemies</div>", unsafe_allow_html=True)
                for enemy in state.get("enemies", []):
                    st.write(f"**{enemy.get('name')}**: {enemy.get('hp')}/{enemy.get('maxHp')} HP | Block: {enemy.get('block')}")
            
            with sub_col2:
                st.markdown("<div class='section-header'>Stats</div>", unsafe_allow_html=True)
                st.write(f"Predicted Incoming: {state.get('predicted_total_damage', 0)}")
                st.write(f"End of Turn Block: {state.get('predicted_end_block', 0)}")

            # Hand View
            render_cards(state.get("hand", []), "Hand")
            
            # Piles View
            p1, p2, p3 = st.columns(3)
            with p1:
                render_cards(state.get("drawPile", []), "Draw Pile")
            with p2:
                render_cards(state.get("discardPile", []), "Discard Pile")
            with p3:
                render_cards(state.get("exhaustPile", []), "Exhaust Pile")
        
        elif state_type == "rewards":
            st.markdown("<div class='section-header'>Rewards</div>", unsafe_allow_html=True)
            for r in state.get("rewards", []):
                st.write(f"- {r.get('description')}")
        
        elif state_type == "map":
            st.markdown("<div class='section-header'>Map Navigation</div>", unsafe_allow_html=True)
            st.write("AI is currently navigating the map.")
        
        else:
            st.write(f"Current State: {state_type}")
            st.json(state, expanded=False)

    with right_col:
        # Visualization Section
        st.markdown("<div class='section-header'>Action Probabilities</div>", unsafe_allow_html=True)
        probs = live_data.get("probs", [])
        if probs:
            # Top 10 indices
            top_indices = np.argsort(probs)[-10:][::-1]
            top_probs = [probs[i] for i in top_indices]
            # Mapping indices to labels
            labels = [f"Act {i}" for i in top_indices]
            
            prob_df = pd.DataFrame({"Probability": top_probs}, index=labels)
            st.bar_chart(prob_df)

        history = get_current_trajectory_history()
        
        # Prepare data for Trend Chart
        chart_data = []
        for s in history:
            chart_data.append({
                "V-Value": s.get("predicted_v", 0.0),
                "Reward": s.get("reward", 0.0)
            })
        
        # Append latest live data point
        if 'v_val' in locals():
            chart_data.append({
                "V-Value": v_val,
                "Reward": live_data.get("reward", 0.0)
            })
        
        if chart_data:
            df_trend = pd.DataFrame(chart_data)
            st.markdown("<div class='section-header'>V-Value & Reward Trend</div>", unsafe_allow_html=True)
            st.line_chart(df_trend)

    # Polling logic
    time.sleep(1)
    st.rerun()

if __name__ == "__main__":
    import numpy as np
    main()
