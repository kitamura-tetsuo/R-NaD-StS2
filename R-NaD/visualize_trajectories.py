import streamlit as st
import json
import os
import pandas as pd
import numpy as np
import datetime
import glob

# Configuration
TRAJECTORY_DIR = "/home/ubuntu/src/R-NaD-StS2/R-NaD/trajectories"
DEFAULT_GAMMA = 0.99

st.set_page_config(page_title="R-NaD Trajectory Visualizer", layout="wide")

def calculate_v_trace(rewards, values, log_probs_bhv, log_probs_pi, terminal, gamma=0.99, clip_rho=1.0, clip_c=1.0):
    """
    Calculate V-trace targets.
    rewards: list of step rewards (length T)
    values: list of predicted V-values (length T)
    log_probs_bhv: log probs of behavior policy (length T)
    log_probs_pi: log probs of current policy (length T)
    terminal: whether the episode ended at the last step
    """
    T = len(rewards)
    vs = np.zeros(T)
    
    # Calculate rho and c
    log_rho = np.array(log_probs_pi) - np.array(log_probs_bhv)
    rho = np.exp(log_rho)
    rho_bar = np.minimum(clip_rho, rho)
    c_bar = np.minimum(clip_c, rho)
    
    # Values of next states
    v_next = np.zeros(T)
    v_next[:-1] = values[1:]
    # If not terminal, we might want to bootstrap, but trajectories in JSON 
    # usually go until terminal or a segment end.
    
    # delta_t = rho_bar * (r_t + gamma * V(x_{t+1}) - V(x_t))
    deltas = rho_bar * (np.array(rewards) + gamma * v_next - np.array(values))
    
    # V-trace recursive formula:
    # v_s = V(x_s) + \delta_s + \gamma * c_s * (v_{s+1} - V(x_{s+1}))
    # worked backwards from T-1 to 0
    curr_diff = 0
    for t in range(T - 1, -1, -1):
        vs[t] = values[t] + deltas[t] + gamma * c_bar[t] * curr_diff
        curr_diff = vs[t] - values[t]
        
    return vs

def calculate_entropy(probs):
    probs = np.array(probs)
    return -np.sum(probs * np.log(probs + 1e-10))

def parse_state_summary(state_json):
    try:
        state = json.loads(state_json)
        summary = {
            "Type": state.get("type", "unknown"),
            "Floor": state.get("floor", 0),
            "HP": 0,
            "Max HP": 0,
            "Gold": state.get("gold", 0),
            "Enemies": 0,
            "Enemy HP": 0
        }
        
        player = state.get("player")
        if player:
            summary["HP"] = player.get("hp", 0)
            summary["Max HP"] = player.get("maxHp", 0)
        else:
            summary["HP"] = state.get("hp", 0)
            summary["Max HP"] = state.get("max_hp", 0)
            
        enemies = state.get("enemies", [])
        summary["Enemies"] = len([e for e in enemies if e.get("hp", 0) > 0])
        summary["Enemy HP"] = sum(e.get("hp", 0) for e in enemies if e.get("hp", 0) > 0)
        
        return summary
    except:
        return {"Type": "Error", "Floor": 0}

st.sidebar.title("R-NaD Trajectory Selector")
files = sorted(glob.glob(os.path.join(TRAJECTORY_DIR, "traj_*.json")), reverse=True)
if not files:
    st.error(f"No trajectory files found in {TRAJECTORY_DIR}")
    st.stop()

selected_file = st.sidebar.selectbox("Select Trajectory", files, format_func=lambda x: os.path.basename(x))

if selected_file:
    with open(selected_file, "r") as f:
        data = json.load(f)
        
    steps = data.get("steps", [])
    if not steps:
        st.warning("Empty trajectory.")
        st.stop()
        
    st.title(f"Trajectory Analysis: {os.path.basename(selected_file)}")
    
    # Process steps
    rewards = [s["reward"] for s in steps]
    v_preds = [s.get("predicted_v", 0.0) for s in steps]
    log_probs_bhv = [s.get("log_prob", 0.0) for s in steps]
    # For visualization, if we don't have log_probs_pi (current policy), 
    # we assume we are visualizing the policy that generated the trajectory (on-policy view)
    # in which case rho = 1.0.
    log_probs_pi = log_probs_bhv # On-policy visualization by default
    
    vs = calculate_v_trace(rewards, v_preds, log_probs_bhv, log_probs_pi, steps[-1]["terminal"], gamma=DEFAULT_GAMMA)
    td_error = np.abs(np.array(vs) - np.array(v_preds))
    cum_rewards = np.cumsum(rewards)
    entropies = [calculate_entropy(s["probs"]) for s in steps]
    
    # State summaries
    summaries = [parse_state_summary(s["state_json"]) for s in steps]
    df_summary = pd.DataFrame(summaries)
    
    # Top Stats Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Steps", len(steps))
    col2.metric("Total Reward", f"{sum(rewards):.2f}")
    col3.metric("Avg TD Error", f"{np.mean(td_error):.4f}")
    col4.metric("Last Floor", df_summary["Floor"].max())
    
    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Charts", "Step Details", "State Analysis", "Raw Data"])
    
    with tab1:
        st.subheader("Values & Targets")
        chart_data = pd.DataFrame({
            "Step": range(len(steps)),
            "Predicted V": v_preds,
            "V-trace Target": vs
        }).set_index("Step")
        st.line_chart(chart_data)
        
        st.subheader("TD Error & Entropy")
        chart_err = pd.DataFrame({
            "Step": range(len(steps)),
            "TD Error": td_error,
            "Entropy": entropies
        }).set_index("Step")
        st.line_chart(chart_err)
        
        st.subheader("Cumulative Reward")
        st.line_chart(pd.Series(cum_rewards, name="Cum Reward"))

    with tab2:
        step_idx = st.slider("Select Step", 0, len(steps) - 1, 0)
        step = steps[step_idx]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.write("**Step Info**")
            st.write(f"Floor: {summaries[step_idx]['Floor']}")
            st.write(f"Type: {summaries[step_idx]['Type']}")
            st.write(f"Action Taken: {step['action_idx']}")
            st.write(f"Reward: {step['reward']:.4f}")
            st.write(f"Predicted V: {v_preds[step_idx]:.4f}")
            st.write(f"V-trace Target: {vs[step_idx]:.4f}")
            st.write(f"TD Error: {td_error[step_idx]:.4f}")
            st.write(f"Entropy: {entropies[step_idx]:.4f}")
            
        with c2:
            st.write("**Policy Probabilities (Top 10)**")
            probs = np.array(step["probs"])
            top_indices = np.argsort(probs)[::-1][:10]
            top_probs = probs[top_indices]
            
            # Simple bar chart for top probs
            prob_df = pd.DataFrame({
                "Action": [str(i) for i in top_indices],
                "Probability": top_probs
            })
            st.bar_chart(prob_df.set_index("Action"))
            
        st.subheader("State JSON")
        st.json(json.loads(step["state_json"]))

    with tab3:
        st.subheader("Critical States Analysis")
        # Flagging logic
        critical_indices = []
        for i, s in enumerate(summaries):
            is_critical = False
            reason = []
            if i == 0 or (i > 0 and summaries[i-1]["Enemies"] == 0 and s["Enemies"] > 0):
                is_critical = True
                reason.append("Combat Start")
            if s["HP"] / (s["Max HP"] + 1e-6) < 0.2:
                is_critical = True
                reason.append("Low HP (<20%)")
            if s["Type"] == "combat" and any(e.get("isBoss") for e in json.loads(steps[i]["state_json"]).get("enemies", [])):
                is_critical = True
                reason.append("Boss Fight")
            
            if is_critical:
                critical_indices.append({
                    "Step": i,
                    "Floor": s["Floor"],
                    "Reason": ", ".join(reason),
                    "V-Pred": v_preds[i],
                    "Target": vs[i],
                    "TD Error": td_error[i]
                })
        
        if critical_indices:
            st.table(pd.DataFrame(critical_indices))
        else:
            st.write("No critical states flagged based on current criteria.")

    with tab4:
        st.write("Raw Step Data")
        st.dataframe(pd.DataFrame(steps).drop(columns=["state_json", "probs", "mask"]))
