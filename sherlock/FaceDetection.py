import cv2
import numpy as np
import sys

facePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(facePath)

cap = cv2.VideoCapture(0)

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
      sub_box = [int(face[0] - face[2]/2),
                                int(face[1] - face[3]/2),
                                int(face[2]*2),
                                int(face[3]*2)
      ]
      # ---- Out of bounds checking
      if sub_box[0] < 0:
        # sub_box[2] += sub_box[0]
        sub_box[0] = 0
      if sub_box[1] < 0:
        # sub_box[3] += sub_box[1]
        sub_box[1] = 0
      if (sub_box[0] + sub_box[2]) > gray_width:
        sub_box[2] = gray_width - sub_box[0] - 1
      if (sub_box[1] + sub_box[3]) > gray_height:
        sub_box[3] = gray_height - sub_box[1] - 1
      bounded_box.append(sub_box)
  else:
    try:
      faces = []
      for i in bounded_box:
        cv2.imshow("heh", gray[i[1]:(i[1] + i[3]), i[0]:(i[0] + i[2])])
        pre_faces = faceCascade.detectMultiScale(
          gray[i[1]:(i[1] + i[3]), i[0]:(i[0] + i[2])],
          scaleFactor= sF,
          minNeighbors=8,
          minSize=(55, 55),
          flags=cv2.CASCADE_SCALE_IMAGE
        )
        faces = faces + np.tolist(pre_faces)
    except AttributeError:
      print("no bounded box")
      faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
      )
  print("break")

  # ---- Draw a rectangle around the faces
  for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

  # ---- Draw a rectangle around the bounded box
  for (x, y, w, h) in bounded_box:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

  cv2.imshow("Face Detector", frame)
  c = cv2.waitKey(7) % 0x100
  if c == 27:
      break

cap.release()
cv2.destroyAllWindows()
