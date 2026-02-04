import cv2
import numpy as np


# ArcFace standard landmark template
ARC_FACE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041],   # right mouth
], dtype=np.float32)


def align_face(image, landmarks, output_size=(112, 112)):
    """
    Align face using 5 landmarks
    landmarks: [(x,y)*5] in original image coords
    """
    src = np.array(landmarks, dtype=np.float32)
    dst = ARC_FACE_LANDMARKS.copy()

    if output_size[0] != 112:
        scale = output_size[0] / 112
        dst *= scale

    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    aligned = cv2.warpAffine(
        image,
        M,
        output_size,
        borderValue=0
    )

    return aligned
