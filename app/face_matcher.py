import numpy as np
import cv2
from app.logger import log_debug


class FaceMatcher:
    """
    FaceMatcher
    -----------
    - Gradient-based embeddings (Sobel magnitude)
    - Cosine similarity
    - Decision threshold controlled centrally
    """

    def __init__(self, threshold=0.55):
        # IMPORTANT: threshold tuned to your real data distribution
        self.threshold = threshold

    def get_embedding(self, face):
        """
        Create a normalized gradient-based embedding
        """
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

    def match(self, live_emb, stored_embs):
        """
        Compare live embedding with stored embeddings
        """
        sims = [self.cosine_similarity(live_emb, e) for e in stored_embs]
        avg_sim = float(np.mean(sims))

        matched = avg_sim >= self.threshold

        log_debug(
            f"Avg cosine similarity: {avg_sim:.3f} | threshold={self.threshold}"
        )

        return matched, avg_sim
