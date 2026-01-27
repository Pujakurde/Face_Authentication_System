import cv2
from app.logger import log_debug

class AntiSpoofingPredictor:
    def __init__(self):
        self.history = []
        self.max_history = 7

    def blur_score(self, gray):
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def temporal_vote(self, label):
        self.history.append(label)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return self.history.count("REAL") >= 4

    def predict(self, face_img, frame_area):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        blur = self.blur_score(gray)
        face_area = face_img.shape[0] * face_img.shape[1]
        size_ratio = face_area / frame_area

        is_real = blur > 80 and size_ratio > 0.04
        final = self.temporal_vote("REAL" if is_real else "FAKE")

        log_debug(f"Blur score: {blur:.2f}")
        log_debug(f"Face/frame size ratio: {size_ratio:.3f}")
        log_debug(f"Anti-spoof decision: {'REAL' if final else 'FAKE'}")

        return ("REAL" if final else "FAKE"), blur
