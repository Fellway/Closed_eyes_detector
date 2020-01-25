import cv2

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
eyeR_cascade = cv2.CascadeClassifier('cascades/haarcascade_righteye_2splits.xml')
eyeL_cascade = cv2.CascadeClassifier('cascades/haarcascade_lefteye_2splits.xml')

cap = cv2.VideoCapture('face3.mp4')

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # faces = face_cascade.detectMultiScale(gray,
    #                                       scaleFactor=1.1,
    #                                       minNeighbors=5,
    #                                       minSize=(30, 30), )

    eyeR = eyeR_cascade.detectMultiScale(gray,
                                         scaleFactor=1.3,
                                         minNeighbors=7,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    eyeL = eyeL_cascade.detectMultiScale(gray,
                                         scaleFactor=1.3,
                                         minNeighbors=7,
                                         minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    if (len(eyeL) == 0) and (len(eyeR) == 0):
        print('no eyes!!!')
    else:
        print('eyes!!!')

    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in eyeR:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in eyeL:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
