import json
import os
import uuid
from datetime import datetime
from pathlib import Path

# Directory to store chat history JSON files
CHATS_DIR = Path(__file__).parent.parent / "data" / "chats"

def ensure_dir_exists():
    if not CHATS_DIR.exists():
        CHATS_DIR.mkdir(parents=True, exist_ok=True)

def generate_chat_id():
    """Generates a unique ID for a new chat session."""
    return str(uuid.uuid4())

def save_chat(chat_id: str, messages: list):
    """
    Saves a list of messages to a JSON file named by chat_id.
    """
    if not messages:
        return
    ensure_dir_exists()
    
    # Extract a title from the first user message (up to 40 chars)
    title = "New Chat"
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            title = content[:40] + ("..." if len(content) > 40 else "")
            break

    chat_data = {
        "id": chat_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": messages
    }
    
    file_path = CHATS_DIR / f"{chat_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)

def get_all_chats() -> list:
    """
    Returns a list of metadata for all saved chats, sorted by updated_at descending.
    """
    ensure_dir_exists()
    chats = []
    
    for file_path in CHATS_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                chats.append({
                    "id": data.get("id"),
                    "title": data.get("title", "Untitled Chat"),
                    "updated_at": data.get("updated_at", "")
                })
        except (json.JSONDecodeError, IOError):
            continue
            
    # Sort by updated_at, newest first
    chats.sort(key=lambda x: x["updated_at"], reverse=True)
    return chats

def load_chat(chat_id: str) -> list:
    """
    Loads the messages list for a specific chat_id.
    Returns an empty list if not found.
    """
    file_path = CHATS_DIR / f"{chat_id}.json"
    if not file_path.exists():
        return []
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("messages", [])
    except (json.JSONDecodeError, IOError):
        return []

def delete_chat(chat_id: str):
    """Deletes a chat file."""
    file_path = CHATS_DIR / f"{chat_id}.json"
    if file_path.exists():
        file_path.unlink()
