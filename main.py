import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

cap = cv2.VideoCapture('video.mp4')

cap.set(3, 1280)
cap.set(4, 720)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL

while True:
    ret, img = cap.read()
    img = cv2.resize(img, (800, 480), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        gray_face = gray[y:y + h, x:x + w]
        face = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 5)

        for (ex, ey, ew, eh) in eyes:
            height = np.size(face, 0)
            width = np.size(face, 1)
            if ey < height / 2.1:
                cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 225, 255), 1)

    cv2.putText(img, 'Time:', (20, 460), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
