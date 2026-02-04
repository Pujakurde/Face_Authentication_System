from cryptography.fernet import Fernet
import os

KEY_FILE = "secret.key"

def load_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
        return key
    else:
        return open(KEY_FILE, "rb").read()

FERNET = Fernet(load_key())

def encrypt_bytes(data: bytes) -> bytes:
    return FERNET.encrypt(data)

def decrypt_bytes(data: bytes) -> bytes:
    return FERNET.decrypt(data)
