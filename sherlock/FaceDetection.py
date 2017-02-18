import cv2
import numpy as np
import sys

facePath = "haarcascade_frontalface_default.xml"
# smilePath = "/usr/local/Cellar/opencv/2.4.7.1/share/OpenCV/haarcascades/haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)
# smileCascade = cv2.CascadeClassifier(smilePath)

cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)

sF = 1.05

bounded_box = []

while True:

    ret, frame = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_width = len(gray[0])
    gray_height = len(gray)

    if bounded_box == []:
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor= sF,
            minNeighbors=8,
            minSize=(55, 55),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for face in faces:
            to_be_added = [int(face[0] - face[2]/2),
                                int(face[1] - face[3]/2),
                                int(face[2]*2),
                                int(face[3]*2)]
            # ---- Out of bounds checking
            if to_be_added[0] < 0:
              to_be_added[2] += to_be_added[0]
              to_be_added[0] = 0
            if to_be_added[1] < 0:
              to_be_added[3] += to_be_added[1]
              to_be_added[1] = 0

            if (to_be_added[0] + to_be_added[2]) > gray_width:
              to_be_added[2] = gray_width - to_be_added[0] - 1
            if (to_be_added[1] + to_be_added[3]) > gray_height:
              to_be_added[3] = gray_height - to_be_added[1] - 1
            bounded_box.append(to_be_added)
    else:
        faces = []
        for i in bounded_box:
            print(i)
            pre_faces = faceCascade.detectMultiScale(
                gray[i[1]:(i[1] + i[3]), i[0]:(i[0] + i[2])],
                scaleFactor= sF,
                minNeighbors=8,
                minSize=(55, 55),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces += pre_faces

#             [] + [[1, 2, 3], [4, 5]]
#             = [[1, 2, 3], [4, 5]]

    # ---- Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
  #     roi_gray = gray[y:y+h, x:x+w]
  #     roi_color = frame[y:y+h, x:x+w]

          # Set region of interest for smiles
  #         for (x, y, w, h) in smile:
  #             print "Found", len(smile), "smiles!"
  #             cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)
  #             #print "!!!!!!!!!!!!!!!!!"

    #cv2.cv.Flip(frame, None, 1)
    bounded_box = []
    cv2.imshow('Face Detector', frame)
    c = cv2.waitKey(7) % 0x100
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()