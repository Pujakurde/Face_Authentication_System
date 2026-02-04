import customtkinter as ctk
import cv2
import numpy as np
import os
import time
import io
from PIL import Image
import logging
import math
from app.crypto_utils import encrypt_bytes, decrypt_bytes
from face_detection.face_detector import FaceDetector
from app.face_matcher import FaceMatcher
from app.welcomescreen import WelcomeScreen
from app.session_store import save_login

# ================= CONFIG ================= #

REGISTER_DIR = "registered_faces"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "app.log")

SAMPLES_REQUIRED = 50
CAPTURE_INTERVAL = 0.25
REGISTER_WINDOW_SECONDS = SAMPLES_REQUIRED * CAPTURE_INTERVAL
REGISTER_MIN_SAMPLES = 30

AUTH_WINDOW = 3
AUTH_VOTES_REQUIRED = 2

REQUIRED_LIVENESS_FRAMES = 5

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

class GuiLogHandler(logging.Handler):
    def __init__(self, gui_logger):
        super().__init__()
        self.gui_logger = gui_logger

    def emit(self, record):
        msg = self.format(record)
        level = record.levelname
        self.gui_logger(msg, level)

# ================= GUI APP ================= #
class FaceAuthApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Secure Face Authentication System")
        self.geometry("900x650")
        self.resizable(False, False)

        # Core
        self.cap = cv2.VideoCapture(0)
        self.detector = FaceDetector()
        self.matcher = FaceMatcher()

        # State
        self.mode = "AUTH"  # AUTH | REGISTER | LOCKED
        self.username = None
        self.flow_state = "WAIT_FACE"

        # Samples Colllecting 
        self.samples = []

        # Authentication Votess
        self.auth_votes = []

        # Authentication
        self.AUTH_VOTE_WINDOW = 7
        self.AUTH_ACCEPT_RATIO = 0.7
        self.auth_finalized = False
        self.auth_candidate_user = None

        # Motion History
        self.motion_history = []

        # Liveness
        self.liveness_frames = 0
        self.liveness_locked = False
        self.liveness_consumed = False
        self.auth_in_progress = False
        self.last_capture_time = 0
        self.liveness_start_time = None
        self.LIVENESS_TIMEOUT = 5  # seconds
        self.register_start_time = None
        self.register_end_time = None
        self.quality_checked = False
        self.quality_passed = False
        self.last_face = None
        self.registration_active = False
        self.system_locked = False

        self.running = True
        self.after_id = None

        # Redirect all logging to GUI
        handler = GuiLogHandler(self.log)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()
        root_logger.addHandler(handler)

        self._build_ui()
        self.log("Application started")

        if self.cap.isOpened():
            self.log("Camera initialized")
        else:
            self.log("Camera failed", "ERROR")

        if not has_registered_users():
            self.log("No registered users found", "WARN")
            self.status.configure(text="No users found. Click Register.")

        self.update_frame()


    # ================= UI ================= #

    def _build_ui(self):

        self.video_label = ctk.CTkLabel(self, text="")
        self.video_label.pack(pady=10)

        self.status = ctk.CTkLabel(self, text="Authenticating… Look at camera", font=("Arial", 16))
        self.status.pack(pady=5)

        btn = ctk.CTkFrame(self)
        btn.pack(pady=5)

        ctk.CTkButton(btn, text="Register", command=self.start_register).pack(side="left", padx=10)
        ctk.CTkButton(btn, text="Exit", command=self.close).pack(side="left", padx=10)

        log_frame = ctk.CTkFrame(self)
        log_frame.pack(fill="both", expand=True, padx=15, pady=10)

        ctk.CTkLabel(log_frame, text="System Logs").pack(anchor="w")

        self.log_box = ctk.CTkTextbox(log_frame, height=200)
        self.log_box.pack(fill="both", expand=True)
        self.log_box.configure(state="disabled")

        for t, c in {
            "INFO": "#EAEAEA",
            "WARN": "#FACC15",
            "ERROR": "#EF4444",
            "SUCCESS": "#22C55E"
        }.items():
            self.log_box.tag_config(t, foreground=c)

    # ================= LOGGING ================= #
    def log(self, msg, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] {msg}\n"

        self.log_box.configure(state="normal")
        self.log_box.insert("end", line, level)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {level} | {msg}\n")

    # ================= CAMERA LOOP ================= #
    def update_frame(self):

        if self.system_locked or not self.running:
            return

        if self.flow_state == "LOCKED":
            return

        ret, frame = self.cap.read()
        if not ret:
            self.schedule_next_frame()
            return

        faces = self.detector.detect(frame)
        if self.auth_in_progress:
            if faces:
                x, y, w, h = faces[0]
                face = frame[y:y+h, x:x+w]
                self.authenticate(face)
            else:
                self.auth_in_progress = False
                self.liveness_locked = False
                self.liveness_consumed = False
                self.flow_state = "WAIT_FACE"
                self.status.configure(text="Face lost")
                self.log("Authentication finished")
            self.render(frame)
            self.schedule_next_frame()
            return

        # ---------- WAIT_FACE ----------
        if self.flow_state == "WAIT_FACE":
            if faces:
                self.flow_state = "WAIT_LIVENESS"
                self.reset_liveness("Blink and move head")
            else:
                self.status.configure(text="Show your face")

        # ---------- WAIT_LIVENESS ----------
        elif self.flow_state == "WAIT_LIVENESS":
            if not faces:
                self.flow_state = "WAIT_FACE"
                self.status.configure(text="Face lost")
            else:
                if not self.liveness_locked and not self.liveness_consumed:
                    landmarks = self.detector.get_raw_landmarks()
                    if landmarks and self.process_liveness(landmarks):
                        self.flow_state = "LIVENESS_PASSED"

        # ---------- LIVENESS_PASSED ----------
        elif self.flow_state == "LIVENESS_PASSED":
            if not faces:
                self.status.configure(text="Show your face")
            else:
                x, y, w, h = faces[0]
                face = frame[y:y+h, x:x+w]

                if self.mode == "REGISTER":
                    if not self.quality_checked:
                        self.quality_checked = True
                        if not self.detector.quality_ok(face):
                            self.quality_passed = False
                            self.log("Registration quality check failed", "WARN")
                            self.status.configure(text="Quality failed. Adjust lighting and try again.")
                            self.abort_registration()
                        else:
                            self.quality_passed = True
                            self.start_registration_window()
                    if self.quality_passed and self.register_start_time:
                        self.update_registration(faces, frame)
                else:
                    if self.liveness_locked and not self.liveness_consumed:
                        self.liveness_consumed = True
                        self.flow_state = "AUTHENTICATING"
                        self.auth_votes.clear()
                        self.log("Entering AUTHENTICATING state")
                        self.status.configure(text="Authenticating… Hold still")

        # ---------- AUTHENTICATING ----------
        elif self.flow_state == "AUTHENTICATING":

            if not faces:
                self.flow_state = "WAIT_FACE"
                self.reset_liveness("Face lost")
                return

            x, y, w, h = faces[0]
            face = frame[y:y+h, x:x+w]

            if not self.auth_in_progress:
                self.auth_in_progress = True
                self.auth_finalized = False
                self.auth_votes.clear()
                self.auth_candidate_user = None
                self.log("Authentication started")

            self.authenticate(face)


        # ---------- RENDER ----------
        
        self.render(frame)
        self.schedule_next_frame()

    def render(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        ctk_img = ctk.CTkImage(img, img, size=(720, 480))
        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img

    # ================= LIVENESS ================= #
    def process_liveness(self, landmarks):
        # If already passed, do NOT re-evaluate
        if self.liveness_locked:
            return True

        blink = self.detector.detect_blink(landmarks)
        head = self.detector.detect_head_movement(landmarks)

        if not blink:
            self.status.configure(text="Please blink")
            return False

        if not head:
            self.status.configure(text="Move head left / right ↔")
            return False

        self.liveness_frames += 1
        self.status.configure(
            text=f"Liveness check {self.liveness_frames}/{REQUIRED_LIVENESS_FRAMES}"
        )

        if self.liveness_frames >= REQUIRED_LIVENESS_FRAMES:
            self.liveness_locked = True
            self.log("Liveness PASSED", "SUCCESS")
            return True

        return False

    def reset_liveness(self, msg=""):
    # CRITICAL: never reset liveness while registering
        if self.mode == "REGISTER" or self.registration_active:
            return

        self.liveness_frames = 0
        self.liveness_locked = False
        self.liveness_consumed = False
        self.auth_in_progress = False
        self.auth_finalized = False
        self.auth_candidate_user = None
        self.detector.reset_liveness()
        self.auth_votes.clear()

        if msg:
            self.status.configure(text=msg)

    # ================= REGISTRATION ================= #
    def start_register(self):

        dialog = ctk.CTkInputDialog(title="Register", text="Enter username")
        name = dialog.get_input()
        if not name:
            return

        name = name.strip().lower()
        path = os.path.join(REGISTER_DIR, name)

        if os.path.exists(path):
            self.status.configure(text="User already exists")
            return

        self.username = name
        self.samples.clear()
        self.auth_votes.clear()
        self.last_face = None

        self.mode = "REGISTER"
        self.flow_state = "WAIT_LIVENESS"   # IMPORTANT
        self.registration_active = True

        self.liveness_frames = 0
        self.liveness_locked = False
        self.liveness_consumed = False
        self.auth_in_progress = False
        self.detector.reset_liveness()
        self.quality_checked = False
        self.quality_passed = False
        self.register_start_time = None
        self.register_end_time = None
        self.last_capture_time = 0

        self.status.configure(text="Blink & move head to register")
        self.log(f"Registration started for '{name}'")

    def start_registration_window(self):
        now = time.time()
        self.register_start_time = now
        self.register_end_time = now + REGISTER_WINDOW_SECONDS
        self.last_capture_time = 0
        self.samples.clear()
        self.status.configure(
            text=f"Registering {self.username}: {len(self.samples)}/{SAMPLES_REQUIRED}"
        )
        self.log("Registration capture window started")

    def update_registration(self, faces, frame):
        now = time.time()
        if self.register_end_time and now >= self.register_end_time:
            self.finish_registration()
            return

        if not faces:
            self.status.configure(text="Keep your face visible")
            return

        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]
        self.last_face = face.copy()

        if now - self.last_capture_time < CAPTURE_INTERVAL:
            return

        if len(self.samples) < SAMPLES_REQUIRED:
            self.last_capture_time = now
            emb = self.matcher.get_embedding(face)
            self.samples.append(emb)
            self.status.configure(
                text=f"Registering {self.username}: {len(self.samples)}/{SAMPLES_REQUIRED}"
            )

    def finish_registration(self):
        if len(self.samples) < REGISTER_MIN_SAMPLES:
            self.log("Registration failed: insufficient samples", "WARN")
            self.status.configure(text="Registration incomplete. Please try again.")
            self.abort_registration()
            return

        user_dir = os.path.join(REGISTER_DIR, self.username)
        os.makedirs(user_dir, exist_ok=True)

        embeddings = np.array(self.samples, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms

        mean_emb = embeddings.mean(axis=0)
        mean_norm = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        dists = 1.0 - np.dot(embeddings, mean_norm)
        dist_mean = float(dists.mean())
        dist_std = float(dists.std())
        keep = dists <= (dist_mean + 2.0 * dist_std)
        if keep.sum() >= max(10, int(0.6 * len(embeddings))):
            embeddings = embeddings[keep]

        buffer = io.BytesIO()
        np.save(buffer, embeddings)
        encrypted = encrypt_bytes(buffer.getvalue())

        with open(os.path.join(user_dir, "embeddings.enc"), "wb") as f:
            f.write(encrypted)

        if self.last_face is not None:
            cv2.imwrite(os.path.join(user_dir, "profile.jpg"), self.last_face)

        self.log(f"User '{self.username}' registered successfully", "SUCCESS")
        self.status.configure(text="Registration complete")

        self.reset_after_registration("Look at camera to authenticate")

    def abort_registration(self):
        self.reset_after_registration("Registration canceled. Look at camera")

    def reset_after_registration(self, msg):
        self.mode = "AUTH"
        self.flow_state = "WAIT_FACE"
        self.registration_active = False
        self.username = None
        self.samples.clear()
        self.auth_votes.clear()
        self.last_face = None
        self.register_start_time = None
        self.register_end_time = None
        self.quality_checked = False
        self.quality_passed = False
        self.liveness_frames = 0
        self.liveness_locked = False
        self.liveness_consumed = False
        self.auth_in_progress = False
        self.last_capture_time = 0
        self.reset_liveness(msg)

    # ================= AUTH ================ #
    def authenticate(self, face):

        if self.auth_finalized:
            return False

        if not has_registered_users():
            self.reset_auth("No registered users")
            return False

        if not self.detector.quality_ok(face):
            self.record_auth_vote(False)
            return False

        live_emb = self.matcher.get_embedding(face)
        self.log("Matching embedding", "DEBUG")

        matched_user = None

        for user in os.listdir(REGISTER_DIR):
            path = os.path.join(REGISTER_DIR, user, "embeddings.enc")
            if not os.path.exists(path):
                continue

            stored = np.load(io.BytesIO(decrypt_bytes(open(path, "rb").read())))
            if stored.shape[0] < REGISTER_MIN_SAMPLES:
                continue

            matched, _ = self.matcher.match(live_emb, stored)
            if matched:
                matched_user = user
                break

        # Enforce same-user consistency
        if matched_user:
            if self.auth_candidate_user is None:
                self.auth_candidate_user = matched_user
            elif matched_user != self.auth_candidate_user:
                matched_user = None

        self.record_auth_vote(bool(matched_user))

        return False
    def record_auth_vote(self, matched):

        self.auth_votes.append(matched)

        if len(self.auth_votes) > self.AUTH_VOTE_WINDOW:
            self.auth_votes.pop(0)

        self.log(
            f"Auth vote {len(self.auth_votes)}/{self.AUTH_VOTE_WINDOW} = "
            f"{'MATCH' if matched else 'NO MATCH'}"
        )

        if len(self.auth_votes) < self.AUTH_VOTE_WINDOW:
            return

        matches = sum(self.auth_votes)
        required = math.ceil(self.AUTH_VOTE_WINDOW * self.AUTH_ACCEPT_RATIO)

        self.auth_votes.clear()
        self.auth_finalized = True

        if self.auth_candidate_user and matches >= required:
            save_login(self.auth_candidate_user)
            self.on_auth_success(self.auth_candidate_user)
        else:
            self.reset_auth("Authentication failed")


    def reset_auth(self, msg):
        self.log(msg, "WARN")
        self.status.configure(text=msg)

        self.auth_in_progress = False
        self.auth_finalized = False
        self.auth_candidate_user = None
        self.auth_votes.clear()

        self.liveness_locked = False
        self.liveness_consumed = False
        self.flow_state = "WAIT_FACE"

    def on_auth_success(self, user):

        self.log(f"Authentication SUCCESS for '{user}'", "SUCCESS")
        self.status.configure(text=f"Welcome {user}")

        # Stop camera + loop FIRST
        self.system_locked = True
        self.running = False

        if self.after_id:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        if self.cap.isOpened():
            self.cap.release()

        # IMPORTANT: do NOT destroy immediately
        self.after_idle(lambda: self._open_welcome(user))

    def _open_welcome(self, user):
        # Destroy auth window safely
        if self.winfo_exists():
            self.destroy()

        # Create NEW root for welcome screen
        welcome = WelcomeScreen(user)
        welcome.mainloop()


    def schedule_next_frame(self):
        if self.running and self.winfo_exists():
            self.after_id = self.after(30, self.update_frame)
    def safe_shutdown(self):
        # Stop update loop
        self.running = False

        # Cancel scheduled after
        if self.after_id:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

        # Release camera
        if self.cap and self.cap.isOpened():
            self.cap.release()

        # Stop Tk cleanly
        try:
            self.quit()
        except Exception:
            pass

    # ================= EXIT ================= #
    def close(self):
        self.log("System shutting down")
        self.safe_shutdown()
        if self.winfo_exists():
            self.destroy()
