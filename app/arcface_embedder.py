import os
import cv2
import numpy as np
import onnxruntime as ort
from app.logger import log_debug


class ArcFaceMatcher:
    """
    ArcFace Matcher (Production-grade)
    ----------------------------------
    - Pretrained ArcFace ONNX (r100)
    - 512-D embeddings
    - Cosine similarity
    """

    def __init__(self, threshold=0.55):
        self.threshold = threshold

        # -------- SAFE ABSOLUTE MODEL PATH --------
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, "models", "arcface_r100.onnx")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"ArcFace model not found at: {MODEL_PATH}")

        self.session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        log_debug("ArcFace model loaded successfully")

    # -------- PREPROCESS --------
    def preprocess(self, face):
        face = cv2.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return face

    # -------- EMBEDDING --------
    def get_embedding(self, face):
        inp = self.preprocess(face)

        emb = self.session.run(
            [self.output_name],
            {self.input_name: inp}
        )[0][0]

        emb = emb / np.linalg.norm(emb)
        return emb.astype(np.float32)

    # -------- MATCHING --------
    def cosine_similarity(self, a, b):
        return float(np.dot(a, b))

    def match(self, live_emb, stored_embs):
        sims = [self.cosine_similarity(live_emb, e) for e in stored_embs]
        avg_sim = float(np.mean(sims))

        log_debug(f"ArcFace similarity: {avg_sim:.3f}")
        return avg_sim
