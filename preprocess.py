import face_recognition
import cv2
import numpy as np

def preprocess(frame: np.ndarray):
    faces = face_recognition.get_locations(frame)
    coords = np.array([face[0] for face in faces])
    mask = np.zeros(frame.shape, dtype=np.uint8)
    cv2.fillPoly(mask, pts=[coords], color=(255, 255, 255))
    return cv2.bitwise_and(frame, mask)