from cryptography.fernet import Fernet
import os

KEY_FILE = "data/secret.key"

def generate_key():
    os.makedirs("data", exist_ok=True)
    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)
    return key

def load_key():
    if not os.path.exists(KEY_FILE):
        return generate_key()
    with open(KEY_FILE, "rb") as f:
        return f.read()

FERNET = Fernet(load_key())

def encrypt_bytes(data: bytes) -> bytes:
    return FERNET.encrypt(data)

def decrypt_bytes(data: bytes) -> bytes:
    return FERNET.decrypt(data)
