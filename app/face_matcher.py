import numpy as np
import cv2
from app.logger import log_debug

class FaceMatcher:
    """
    FaceMatcher
    -----------
    - Gradient-based embeddings
    - Cosine similarity
    - NO decision logic here
    """

    def __init__(self):
        pass

    def get_embedding(self, face):
        face = cv2.resize(face, (112, 112))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        mag = cv2.magnitude(gx, gy)
        emb = cv2.resize(mag, (16, 16)).flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-8)

        return emb

    def cosine_similarity(self, a, b):
        return float(np.dot(a, b))

    def match(self, live_emb, stored_embs, threshold=0.5):
        scores = np.dot(stored_embs, live_emb)
        score = float(np.mean(scores))
        matched = score >= threshold
        return matched, score

