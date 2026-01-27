'''import cv2
import mediapipe as mp
from app.gui import open_gui_and_check_user
from app.register import ask_username, register_face

face_history = []
HISTORY_SIZE = 7
motion_history = []
MOTION_HISTORY_SIZE = 7
MOTION_THRESHOLD = 8  # pixels
spoof_history = []
SPOOF_HISTORY_SIZE = 7
BLUR_THRESHOLD = 80        # tune 70–120 if needed
MIN_FACE_RATIO = 0.04     # 4% of frame

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Camera not opening")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    # MediaPipe face detection
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    authenticated_face = None
    stop_camera = False


    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        h, w, _ = frame.shape
        if not results.detections:
            motion_history.clear()
            face_history.clear()
            spoof_history.clear()


        # Draw face boxes
        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                center_x = x + bw // 2
                center_y = y + bh // 2
                motion_history.append((center_x, center_y))
                if len(motion_history) > MOTION_HISTORY_SIZE:
                    motion_history.pop(0)
                is_live = False

                if len(motion_history) >= 2:
                    dx = motion_history[-1][0] - motion_history[0][0]
                    dy = motion_history[-1][1] - motion_history[0][1]
                    movement = (dx ** 2 + dy ** 2) ** 0.5
                    is_live = movement > (0.03 * bw)


                x = max(0, x)
                y = max(0, y)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 255, 0),
                    2
                )

                face_roi = frame[y:y + bh, x:x + bw]
                # ---- Anti-spoof signals ----
                gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()

                frame_area = frame.shape[0] * frame.shape[1]
                face_area = bw * bh
                size_ratio = face_area / frame_area

                is_not_spoof = (blur_score > BLUR_THRESHOLD) and (size_ratio > MIN_FACE_RATIO)

                spoof_history.append(is_not_spoof)
                if len(spoof_history) > SPOOF_HISTORY_SIZE:
                    spoof_history.pop(0)

                stable_real_spoof = spoof_history.count(True) >= 4


                is_real_face = False
                if face_roi.size != 0:
                    rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    mesh_result = face_mesh.process(rgb_face)
                    is_real_face = mesh_result.multi_face_landmarks is not None

                face_history.append(is_real_face)
                if len(face_history) > HISTORY_SIZE:
                    face_history.pop(0)
                

                stable_real = face_history.count(True) >= 4

                if stable_real and is_live and stable_real_spoof:
                    label = "AUTHENTIC LIVE FACE"
                    color = (0, 255, 0)

                    authenticated_face = face_roi.copy()
                    stop_camera = True
                    break

                elif stable_real and not is_live:
                    label = "FACE NOT LIVE"
                    color = (0, 255, 255)

                elif stable_real and is_live and not stable_real_spoof:
                    label = "SPOOF DETECTED"
                    color = (0, 0, 255)
                
                else:
                    label = "NOT A FACE"
                    color = (0, 0, 255)
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        if stop_camera:
            break
        # Show window
        cv2.imshow("Face Authentication - Baseline", frame)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ===== CAMERA LOOP ENDS HERE =====
    cap.release()
    cv2.destroyAllWindows()

    if authenticated_face is not None:
        is_registered = False
        username = None

        result = open_gui_and_check_user(is_registered, username)

        if result == "GRANTED":
            print("ACCESS GRANTED")

        elif result == "REGISTER":
            username = ask_username()
            if username:
                path = register_face(authenticated_face, username)
                print(f"USER REGISTERED: {username}")
                print(f"Saved at: {path}")
            else:
                print("REGISTRATION CANCELLED")

        elif result == "EXIT":
            print("EXIT")


        

if __name__ == "__main__":
    main()'''
from app.gui import FaceAuthApp

if __name__ == "__main__":
    app = FaceAuthApp()
    app.mainloop()

