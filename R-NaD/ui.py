import streamlit as st
import requests

st.set_page_config(page_title="R-NaD Control Panel", layout="centered")

st.title("R-NaD AI Control Panel")

st.write("This panel controls the R-NaD learning process in the Slay the Spire 2 Mod.")

# Configuration
SERVER_URL = "http://127.0.0.1:8081"

def get_status():
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=1)
        if response.status_code == 200:
            return response.json().get("learning_active", False)
    except requests.exceptions.RequestException:
        return None
    return None

def toggle_learning(start: bool):
    endpoint = "/start" if start else "/stop"
    try:
        response = requests.get(f"{SERVER_URL}{endpoint}")
        if response.status_code == 200:
            st.success("Learning started!" if start else "Learning stopped!")
        else:
            st.error(f"Failed to change state. Server responded with {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to communicate with Godot Python Backend: {e}")

def start_new_game():
    try:
        response = requests.get(f"{SERVER_URL}/new_game")
        if response.status_code == 200:
            st.success("New game command sent to Slay the Spire 2!")
        else:
            st.error(f"Failed to send command. Server responded with {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to communicate with Godot Python Backend: {e}")

# Fetch current status
current_status = get_status()

st.divider()

if current_status is None:
    st.error("Cannot connect to Godot backend. Ensure the Slay the Spire 2 mod is running.")
else:
    st.markdown(f"**Current Status:** {'🟢 Learning ACTIVE' if current_status else '🔴 Learning INACTIVE'}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Learning", disabled=current_status, use_container_width=True, type="primary"):
            toggle_learning(True)
            st.rerun()
            
    with col2:
        if st.button("Stop Learning", disabled=not current_status, use_container_width=True):
            toggle_learning(False)
            st.rerun()

    st.divider()
    if st.button("🚀 Start New Game", use_container_width=True):
        start_new_game()

st.divider()
st.info("The learning state is communicated to the running Godot instance via HTTP.")
