import os
import httpx
import json
from pathlib import Path
from dotenv import load_dotenv

# Existing human play history location
REPLAY_DIR = Path("/mnt/nas/StS2/replay")
CHANNEL_ID = "1491584922530484375" # record channel

def download_human_data():
    """
    Downloads human_play_*.jsonl files from the Discord 'record' channel
    and saves them to the REPLAY_DIR if they don't already exist.
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

    try:
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
                            with open(target_path, "wb") as f:
                                f.write(file_res.content)
                            download_count += 1
                        else:
                            print(f"Failed to download {filename}: {file_res.status_code}")

    except Exception as e:
        print(f"Error during download: {e}")

    print(f"Finished. Downloaded: {download_count}, Skipped: {skip_count}")

if __name__ == "__main__":
    download_human_data()
