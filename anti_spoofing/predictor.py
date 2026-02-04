import cv2
import numpy as np
from app.logger import log_debug


class AntiSpoofingPredictor:
    def __init__(self):
        self.real_frames = 0
        self.prev_gray = None

        self.BLUR_THRESHOLD = 80
        self.MOTION_THRESHOLD = 2.0
        self.MIN_FACE_RATIO = 0.04
        self.REQUIRED_REAL_FRAMES = 3
  # anti-spoof internal stability

    def blur_score(self, gray):
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def motion_score(self, gray):
        # FIXED SIZE FOR MOTION (PRODUCTION SAFE)
        gray = cv2.resize(gray, (128, 128))

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0

        # shape now ALWAYS matches
        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        return float(np.mean(diff))

    def predict(self, frame, face_bbox):
        """
        frame: full BGR frame
        face_bbox: (x, y, w, h)
        """

        x, y, w, h = face_bbox

        # -------- Safety checks --------
        if frame is None or w <= 0 or h <= 0:
            self.reset()
            return False

        # -------- Convert to gray --------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------- Motion score --------
        motion = self.motion_score(gray)

        # -------- Blur score --------
        face = gray[y:y+h, x:x+w]
        if face.size == 0:
            return False

        blur = cv2.Laplacian(face, cv2.CV_64F).var()

        # -------- Face / frame ratio --------
        face_area = w * h
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_h * frame_w   #  FIX HERE

        ratio = face_area / frame_area

        # -------- Decision thresholds --------
        real = (
            blur > self.BLUR_THRESHOLD and
            motion > self.MOTION_THRESHOLD and
            ratio > self.MIN_FACE_RATIO
        )

        if real:
            self.real_frames += 1
        else:
            self.real_frames = 0

        decision = self.real_frames >= self.REQUIRED_REAL_FRAMES

        # -------- Debug --------
        print(
            f"[AntiSpoof] blur={blur:.1f} "
            f"motion={motion:.2f} "
            f"ratio={ratio:.3f} "
            f"real_frames={self.real_frames}/{self.REQUIRED_REAL_FRAMES} "
            f"DECISION={'REAL' if decision else 'FAKE'}"
        )

        return decision
    def reset(self):
        self.real_frames = 0
        self.prev_gray = None

