import face_recognition
import cv2
import numpy as np

def preprocess(frame: np.ndarray):
    faces = face_recognition.face_locations(frame, model="cnn")
    mask = np.zeros(frame.shape, dtype=np.uint8)
    for (top, right, bottom, left) in faces:
      cv2.fillPoly(mask, pts=[np.array([[top, left], [bottom, right]])], color=(255, 255, 255))
    return cv2.bitwise_and(frame, mask)