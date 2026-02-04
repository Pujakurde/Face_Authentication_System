import cv2
import mediapipe as mp
import numpy as np
from app.logger import log_debug, log_warn

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


class FaceDetector:
    """
    FaceDetector (Production-safe)
    ------------------------------
    - Face detection (MediaPipe)
    - Raw landmark extraction (468 points)
    - Quality checks
    - Liveness primitives (blink + head move)
    """

    def __init__(self):
        self.detector = mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.6
        )

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.results = None

        # Liveness state
        self.prev_nose_x = None
        self.blink_counter = 0
        self.blinked = False
        self.head_moved = False


    # ================= FACE DETECTION ================= #
    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = self.detector.process(rgb)
        self.results = self.face_mesh.process(rgb)

        h, w, _ = frame.shape
        faces = []

        if not detections.detections:
            log_warn("No face detected")
            return faces

        for det in detections.detections:
            box = det.location_data.relative_bounding_box
            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            faces.append((max(0, x), max(0, y), bw, bh))
            log_debug(f"Face detected x={x}, y={y}, w={bw}, h={bh}")

        return faces


    # ================= LANDMARK VALID ================= #
    def valid_landmarks(self):
        return (
            self.results is not None and
            self.results.multi_face_landmarks is not None
        )


    # ================= RAW LANDMARKS (468) ================= #
    def get_raw_landmarks(self):
        """
        Returns MediaPipe NormalizedLandmark list (468 points).
        REQUIRED for blink & head movement.
        """
        if not self.valid_landmarks():
            return None
        return self.results.multi_face_landmarks[0].landmark


    # ================= QUALITY CHECK ================= #
    def quality_ok(self, face):
        h, w = face.shape[:2]
        if h < 120 or w < 120:
            return False

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 70:
            return False

        mean_intensity = gray.mean()
        if mean_intensity < 60 or mean_intensity > 200:
            return False

        return True


    # ================= BLINK DETECTION ================= #
    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)


    def detect_blink(self, landmarks):
        if landmarks is None or len(landmarks) < 468:
            return False

        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        def eye_points(indexes):
            return np.array([[landmarks[i].x, landmarks[i].y] for i in indexes])

        left_eye = eye_points(LEFT_EYE)
        right_eye = eye_points(RIGHT_EYE)

        ear = (self.eye_aspect_ratio(left_eye) +
               self.eye_aspect_ratio(right_eye)) / 2.0

        if ear < 0.20:
            self.blink_counter += 1
        else:
            if self.blink_counter >= 2:
                self.blinked = True
            self.blink_counter = 0

        return self.blinked


    # ================= HEAD MOVEMENT ================= #
    def detect_head_movement(self, landmarks):
        if landmarks is None or len(landmarks) < 468:
            return False

        nose_x = landmarks[1].x  # Nose tip

        if self.prev_nose_x is None:
            self.prev_nose_x = nose_x
            return False

        movement = abs(nose_x - self.prev_nose_x)
        self.prev_nose_x = nose_x

        if movement > 0.02:
            self.head_moved = True

        return self.head_moved


    # ================= RESET ================= #
    def reset_liveness(self):
        self.prev_nose_x = None
        self.blink_counter = 0
        self.blinked = False
        self.head_moved = False
