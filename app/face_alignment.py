import cv2
import numpy as np

def align_face(image, landmarks, output_size=(112, 112)):
    """
    Align face using 5-point landmarks (ArcFace style)
    landmarks = [left_eye, right_eye, nose, left_mouth, right_mouth]
    """

    src = np.array(landmarks, dtype=np.float32)

    # Standard ArcFace reference points
    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    M, _ = cv2.estimateAffinePartial2D(src, dst)
    aligned = cv2.warpAffine(image, M, output_size)

    return aligned
