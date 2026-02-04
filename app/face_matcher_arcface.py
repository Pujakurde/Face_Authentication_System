import cv2
import numpy as np
import onnxruntime as ort
from app.logger import log_debug
import os


class ArcFaceMatcher:
    """
    ArcFace Matcher (Production-grade)
    ----------------------------------
    - Pretrained ArcFace R100 ONNX
    - 512-D embeddings
    - Cosine similarity
    """

    def __init__(self, model_path="models/arcface_r100.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace model not found at: {model_path}")

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, face):
        face = cv2.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return face

    def get_embedding(self, face):
        inp = self.preprocess(face)
        emb = self.session.run(
            [self.output_name],
            {self.input_name: inp}
        )[0][0]

        emb = emb / np.linalg.norm(emb)
        return emb.astype(np.float32)

    def match(self, live_emb, stored_embs):
        sims = np.dot(stored_embs, live_emb)
        avg_sim = float(np.mean(sims))

        log_debug(f"ArcFace similarity: {avg_sim:.3f}")
        return avg_sim
