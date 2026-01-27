import cv2
import mediapipe as mp
from app.logger import log_debug, log_warn

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


class FaceDetector:
    def __init__(self):
        self.detector = mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        h, w, _ = frame.shape
        faces = []

        if not results.detections:
            log_warn("No face detected in frame")
            return faces

        for det in results.detections:
            box = det.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            x, y = max(0, x), max(0, y)
            faces.append((x, y, bw, bh))
            log_debug(f"Face detected at x={x}, y={y}, w={bw}, h={bh}")

        return faces

    def valid_landmarks(self, frame, bbox):
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            log_warn("Empty face crop")
            return False

        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)

        valid = result.multi_face_landmarks is not None
        log_debug(f"Face landmarks valid = {valid}")

        return valid

    def quality_ok(self, face):
        h, w = face.shape[:2]

        # size check
        if h < 120 or w < 120:
            return False

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # blur check
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 70:
            return False

        # brightness check
        mean_intensity = gray.mean()
        if mean_intensity < 60 or mean_intensity > 200:
            return False

        return True

