from __future__ import annotations

from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


def extract_face_landmarks(image: np.ndarray) -> List[Tuple[int, int]] | None:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape

    points: List[Tuple[int, int]] = []
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points.append((x, y))

    return points
