import os
import httpx
import json
from pathlib import Path
from dotenv import load_dotenv

# Existing human play history location
REPLAY_DIR = Path("/mnt/nas/StS2/replay")
CHANNEL_ID = "1491584922530484375" # record channel

def cleanup_manual_retires(file_path: Path):
    """
    Checks the last line of a human_play_*.jsonl file.
    If it's a non-victory game_over state that appears to be a manual retire
    (e.g., reason is main_menu or the player still had HP > 0), 
    removes that terminal step to avoid negative reward penalty during training.
    """
    if not file_path.exists():
        return

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        if not lines:
            return

        last_line = lines[-1].strip()
        if not last_line:
            # Try second to last if last is empty
            if len(lines) > 1:
                last_line = lines[-2].strip()
            else:
                return

        try:
            data = json.loads(last_line)
            # Check if this is a game_over state
            state = data.get("state", {})
            if state.get("type") == "game_over":
                victory = state.get("victory", False)
                reason = state.get("reason", "")
                
                # Check for HP to confirm it's not a real death
                player = state.get("player", {})
                hp = player.get("hp", 1) # Default to 1 if not found to be safe
                
                is_manual_retire = False
                if not victory:
                    if reason == "main_menu":
                        is_manual_retire = True
                    elif hp > 0:
                        # If victory is false but hp > 0, it's likely a manual retire/abandon
                        is_manual_retire = True
                
                if is_manual_retire:
                    print(f"Manual retire detected in {file_path.name} (reason: {reason}, hp: {hp}). Removing terminal step.")
                    # Remove the last line (the game_over state)
                    # We keep the lines before it.
                    with open(file_path, "w") as f:
                        f.writelines(lines[:-1])
                        
        except json.JSONDecodeError:
            pass
            
    except Exception as e:
        print(f"Error cleaning up {file_path.name}: {e}")

def download_human_data():
    """
    Downloads human_play_*.jsonl files from the Discord 'record' channel
    and saves them to the REPLAY_DIR if they don't already exist.
    Also cleans up manual retires in the downloaded files.
    """
    # Load .env from root or parent directory
    # Based on the research, .env is in the project root.
    # Current file is in R-NaD/, so look in parent.
    r_nad_dir = Path(__file__).parent.absolute()
    root_dir = r_nad_dir.parent
    env_path = root_dir / ".env"
    
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(f"Warning: .env not found at {env_path}")

    token = os.getenv("DISVORD_BOT_TOKEN")
    if not token:
        print("Error: DISVORD_BOT_TOKEN not found in environment or .env")
        return

    if not REPLAY_DIR.exists():
        print(f"Creating directory: {REPLAY_DIR}")
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": f"Bot {token}",
        "User-Agent": "AutoSlayer-Data-Downloader (1.0)"
    }

    base_url = f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages"
    
    print(f"Fetching messages from Discord channel {CHANNEL_ID}...")
    
    download_count = 0
    skip_count = 0
    cleanup_count = 0

    try:
        # First, run cleanup on existing files in REPLAY_DIR
        for p in REPLAY_DIR.glob("human_play_*.jsonl"):
            cleanup_manual_retires(p)

        with httpx.Client(timeout=30.0) as client:
            # We fetch recent messages. For a more robust solution, we could use pagination (before=ID).
            # For now, fetching the last 100 messages should covers recent uploads.
            response = client.get(base_url, headers=headers, params={"limit": 100})
            
            if response.status_code != 200:
                print(f"Failed to fetch messages: {response.status_code} {response.text}")
                return

            messages = response.json()
            for msg in messages:
                attachments = msg.get("attachments", [])
                for att in attachments:
                    filename = att.get("filename", "")
                    if filename.startswith("human_play_") and filename.endswith(".jsonl"):
                        target_path = REPLAY_DIR / filename
                        if target_path.exists():
                            # Skip if already downloaded
                            skip_count += 1
                            continue
                        
                        # Download file
                        url = att.get("url")
                        print(f"Downloading {filename}...")
                        file_res = client.get(url)
                        if file_res.status_code == 200:
                            content = file_res.content
                            with open(target_path, "wb") as f:
                                f.write(content)
                            
                            # Clean up after download
                            cleanup_manual_retires(target_path)
                            download_count += 1
                        else:
                            print(f"Failed to download {filename}: {file_res.status_code}")

    except Exception as e:
        print(f"Error during download: {e}")

    print(f"Finished. Downloaded: {download_count}, Skipped: {skip_count}")

if __name__ == "__main__":
    download_human_data()
