import face_recognition
import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split

def preprocess(frame: np.ndarray):
    faces = face_recognition.face_locations(frame, model="cnn")
    mask = np.zeros(frame.shape, dtype=np.uint8)
    for (top, right, bottom, left) in faces:
      cv2.fillPoly(mask, pts=[np.array([[top, left], [bottom, right]])], color=(255, 255, 255))
    return cv2.bitwise_and(frame, mask)

def get_data():
    fake = glob.glob("./dataset/fake/*")
    real = glob.glob("./dataset/real/*")
    y = [0]*len(fake) + [1]*len(real)
    X = fake + real
    print(f"Fake videos: {len(fake)} | Real Videos: {len(real)} | Total Videos: {len(y)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=18)
    return X_train, X_test, y_train, y_test

def get_current(X, y):
    random_indices = np.random.randint(low=0, high=len(X)-1, size=130)
    X = [X[idx] for idx in random_indices]
    y = [y[idx] for idx in random_indices]
    return train_test_split(X, y, test_size=0.2, shuffle=True, random_state=12)