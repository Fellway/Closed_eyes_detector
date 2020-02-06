import numpy as np
import cv2
import time
import pygame


# This function is only used for the trackbar
def nothing(x):
    pass


# colors
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

# cascades
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

# cv2 settings
cv2.namedWindow('main')
cv2.namedWindow('right_eye')
cv2.namedWindow('left_eye')
cv2.namedWindow('blob')
cv2.namedWindow('face')
cv2.moveWindow('right_eye', 200, 20)
cv2.moveWindow('left_eye', 400, 20)
cv2.moveWindow('blob', 600, 20)
cv2.moveWindow('face', 400, 200)
cv2.createTrackbar('threshold', 'main', 90, 255, nothing)

# blob detector settings
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)

# pupils format settings
erode = 2
dilate = 4
blur = 5

# other settings
ALARM_SOUND_PATH = 'media/beep.mp3'
VIDEO_PATH = 'media/me.mp4'
pygame.init()
alarm_time = 2
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


def format_pupils(eye_image, threshold):
    gray_frame = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    _, pupil = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    pupil = cv2.erode(pupil, None, iterations=erode)
    pupil = cv2.dilate(pupil, None, iterations=dilate)
    pupil = cv2.medianBlur(pupil, blur)
    cv2.imshow('blob', pupil)
    return detector.detect(pupil)


def alarm(time_initial):
    time_counter = time.time() - time_initial
    if time_counter > alarm_time:
        pygame.mixer.music.load(ALARM_SOUND_PATH)
        pygame.mixer.music.play(1)


def detect_eyes(face):
    left_eye = None
    right_eye = None
    gray_detected_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_detected_face, 1.3, 10)
    height = np.size(face, 0)
    width = np.size(face, 1)
    for (xx, yy, ww, hh) in eyes:
        if yy > height / 2.2:
            pass
        else:
            eye_center = xx + ww / 2
            if eye_center < width * 0.5:
                left_eye = face[yy:yy + hh, xx:xx + ww]
                cv2.imshow('left_eye', left_eye)
                cv2.rectangle(face, (xx, yy), (xx + ww, yy + hh), GREEN, 1)
            else:
                right_eye = face[yy:yy + hh, xx:xx + ww]
                cv2.imshow('right_eye', right_eye)
                cv2.rectangle(face, (xx, yy), (xx + ww, yy + hh), GREEN, 1)
    return left_eye, right_eye


def main():
    time_initial = time.time()
    cap = cv2.VideoCapture(VIDEO_PATH)
    while True:
        threshold = cv2.getTrackbarPos('threshold', 'main')
        _, img = cap.read()
        gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_face, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), BLUE, 1)
            detected_face = img[y:y + h, x:x + w]
            cv2.imshow("face", detected_face)
            detected_eyes = detect_eyes(detected_face)
            for eye in detected_eyes:
                if eye is not None:
                    pupils = format_pupils(eye, threshold)
                    cv2.drawKeypoints(eye, pupils, eye, color=RED, flags=0)
                    if not pupils:
                        alarm(time_initial)
                    else:
                        time_initial = time.time()
                else:
                    alarm(time_initial)
        cv2.imshow('main', img)

        key = cv2.waitKey(20)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
