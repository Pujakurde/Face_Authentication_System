import json
import os
from datetime import datetime

def meta_path(username):
    return os.path.join("registered_faces", username, "meta.json")

def save_login(username):
    path = meta_path(username)
    data = {
        "last_login": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_last_login(username):
    path = meta_path(username)
    if not os.path.exists(path):
        return "First login"
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("last_login", "Unknown")
