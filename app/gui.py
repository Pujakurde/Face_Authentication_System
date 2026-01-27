import customtkinter as ctk
import cv2
import numpy as np
import os
import time
import io
from PIL import Image
import tkinter.messagebox as messagebox
from app.crypto_utils import encrypt_bytes, decrypt_bytes
from face_detection.face_detector import FaceDetector
from anti_spoofing.predictor import AntiSpoofingPredictor
from app.face_matcher import FaceMatcher


# ================= CONFIG ================= #

REGISTER_DIR = "registered_faces"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

SAMPLES_REQUIRED = 50
CAPTURE_INTERVAL = 0.25
AUTH_VOTES_REQUIRED = 3
AUTH_WINDOW = 6



os.makedirs(REGISTER_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ================= UTIL ================= #

def has_registered_users():
    for user in os.listdir(REGISTER_DIR):
        if os.path.exists(os.path.join(REGISTER_DIR, user, "embeddings.enc")):
            return True
    return False


# ================= GUI APP ================= #

class FaceAuthApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Secure Face Authentication System")
        self.geometry("900x650")
        self.resizable(False, False)

        # -------- Core -------- #
        self.cap = cv2.VideoCapture(0)
        self.detector = FaceDetector()
        self.matcher = FaceMatcher()
        self.spoof = AntiSpoofingPredictor()

        # -------- State -------- #
        self.mode = "AUTH"        # AUTH | REGISTER | LOCKED
        self.username = None
        self.samples = []
        self.auth_votes = []
        self.last_capture_time = 0
        self.waiting_for_choice = False

        # -------- UI -------- #
        self._build_ui()

        # -------- Logs -------- #
        self.log("Application started")
        if self.cap.isOpened():
            self.log("Camera initialized")
        else:
            self.log("Camera failed to initialize", "ERROR")

        if not has_registered_users():
            self.log("No registered users found. Authentication only mode.", "WARN")
            self.status.configure(text="No users found. Click Register to enroll.")

        self.update_frame()


    # ================= UI ================= #

    def _build_ui(self):

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=10)

        self.status = ctk.CTkLabel(
            self,
            text="Authenticating… Look at the camera",
            font=("Arial", 16)
        )
        self.status.pack(pady=5)

        # Buttons
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=5)

        ctk.CTkButton(
            btn_frame, text="Register", command=self.start_register
        ).pack(side="left", padx=10)

        ctk.CTkButton(
            btn_frame, text="Exit", command=self.close
        ).pack(side="left", padx=10)

        # -------- LOG BUS -------- #
        log_frame = ctk.CTkFrame(self)
        log_frame.pack(fill="both", expand=True, padx=15, pady=10)

        ctk.CTkLabel(log_frame, text="System Logs").pack(anchor="w")

        self.log_box = ctk.CTkTextbox(log_frame, height=200)
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

        self.log_box.tag_config("INFO", foreground="#EAEAEA")
        self.log_box.tag_config("WARN", foreground="#FACC15")
        self.log_box.tag_config("ERROR", foreground="#EF4444")
        self.log_box.tag_config("SUCCESS", foreground="#22C55E")


    # ================= LOGGING ================= #

    def log(self, message, level="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}\n"

        self.log_box.configure(state="normal")
        self.log_box.insert("end", line, level)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {level} | {message}\n"
            )


    # ================= CAMERA LOOP ================= #

    def update_frame(self):

        if self.mode == "LOCKED":
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(50, self.update_frame)
            return

        faces = self.detector.detect(frame)

        if faces:
            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]

            if self.detector.valid_landmarks(frame, (x, y, w, h)):
                spoof_ok = self.spoof.predict(
                    face, frame.shape[0] * frame.shape[1]
                )

                if spoof_ok:
                    if self.mode == "REGISTER":
                        self.collect_sample(face)
                    else:
                        self.authenticate(face)
                else:
                    self.status.configure(text="Spoof detected. Use a real face.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(img, img, size=(720, 480))

        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img

        self.after(30, self.update_frame)


    # ================= REGISTRATION ================= #

    def start_register(self):

        dialog = ctk.CTkInputDialog(title="Register", text="Enter username")
        name = dialog.get_input()

        if not name:
            self.log("Registration cancelled", "WARN")
            return

        name = name.strip()
        user_dir = os.path.join(REGISTER_DIR, name)

        if os.path.exists(user_dir):
            self.log(f"User '{name}' already exists", "WARN")
            self.status.configure(text="User already exists")
            return

        self.username = name.strip().lower()

        self.samples.clear()
        self.mode = "REGISTER"

        self.log(f"Registration started for '{self.username}'")
        self.status.configure(
            text=f"Registering {self.username}: 0/{SAMPLES_REQUIRED}"
        )


    def collect_sample(self, face):

        if not self.username:
            self.log("Collect sample called without username", "ERROR")
            self.mode = "AUTH"
            return

        now = time.time()
        if now - self.last_capture_time < CAPTURE_INTERVAL:
            return

        self.last_capture_time = now

        if face.shape[0] < 120 or face.shape[1] < 120:
            self.log("Face too small for registration", "WARN")
            return

        emb = self.matcher.get_embedding(face)
        self.samples.append(emb)

        self.status.configure(
            text=f"Registering {self.username}: {len(self.samples)}/{SAMPLES_REQUIRED}"
        )

        if len(self.samples) >= SAMPLES_REQUIRED:
            user_dir = os.path.join(REGISTER_DIR, self.username)
            os.makedirs(user_dir, exist_ok=True)

            buffer = io.BytesIO()
            np.save(buffer, np.array(self.samples, dtype=np.float32))
            encrypted = encrypt_bytes(buffer.getvalue())

            with open(os.path.join(user_dir, "embeddings.enc"), "wb") as f:
                f.write(encrypted)

            self.log(f"User '{self.username}' registered successfully", "SUCCESS")
            self.status.configure(text="Registration complete")
            self.mode = "AUTH"
            self.username = None


    # ================= AUTHENTICATION ================= #
    def authenticate(self, face):

        if not has_registered_users():
            self.status.configure(text="No registered users. Please register.")
            return

        live_emb = self.matcher.get_embedding(face)

        # initialize per-user tracker
        if not hasattr(self, "_current_user"):
            self._current_user = None

        for user in os.listdir(REGISTER_DIR):

            # reset votes when switching user
            if self._current_user != user:
                self.auth_votes.clear()
                self._current_user = user

            path = os.path.join(REGISTER_DIR, user, "embeddings.enc")
            if not os.path.exists(path):
                continue

            # decrypt embeddings
            with open(path, "rb") as f:
                encrypted = f.read()

            stored = np.load(io.BytesIO(decrypt_bytes(encrypted)))

            matched, score = self.matcher.match(live_emb, stored)

            self.auth_votes.append(matched)
            if len(self.auth_votes) > AUTH_WINDOW:
                self.auth_votes.pop(0)

            self.log(f"Matching '{user}' | score={score:.2f}")

            # ---- SUCCESS ----
            if self.auth_votes.count(True) >= AUTH_VOTES_REQUIRED:
                self.log(f"Authentication SUCCESS for '{user}'", "SUCCESS")
                self.status.configure(text=f"Access Granted – {user}")
                self.mode = "LOCKED"
                self.cap.release()
                return

        # ---- WAIT until window fills ----
        if len(self.auth_votes) < AUTH_WINDOW:
            self.status.configure(text="Authenticating… Hold still")
            return

        # ---- FINAL FAIL ----
        self.log("Authentication failed after vote window", "WARN")
        self.status.configure(text="Access Denied")
        self.auth_votes.clear()

        


    # ================= EXIT ================= #

    def close(self):
        self.log("System shutting down")
        self.cap.release()
        self.destroy()
