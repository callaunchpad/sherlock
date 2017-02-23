import cv2
import numpy as np

facePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(facePath)

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

sF = 1.05
mN = 20

past_first_frame = False
new_face = 0

while True:
	ret, frame = cap.read()
	img = frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	gray_width = len(gray[0])
	gray_height = len(gray)

	bounded_box = []

	if (not bounded_box) or (not new_face%20):
		faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=sF,
		minNeighbors=mN,
		minSize=(55, 55),
		flags=cv2.CASCADE_SCALE_IMAGE
		)
		for face in faces:
			sub_box = [int(face[0] - face[2]/4),
						int(face[1] - face[3]/4),
						int(face[2]*1.5),
						int(face[3]*1.5)
			]
			# ---- Out of bounds checking
			if sub_box[0] < 0:
				sub_box[0] = 0
			if sub_box[1] < 0:
				sub_box[1] = 0
			if (sub_box[0] + sub_box[2]) > gray_width:
				sub_box[2] = gray_width - sub_box[0] - 1
			if (sub_box[1] + sub_box[3]) > gray_height:
				sub_box[3] = gray_height - sub_box[1] - 1
			bounded_box.append(sub_box)
	else:
		past_first_frame = True
		faces = []
		for i in bounded_box:
			adjustments = [i[0], i[1]]
			mini_gray = gray[i[1]:(i[1] + i[3]), i[0]:(i[0] + i[2])]
			pre_faces = faceCascade.detectMultiScale(
				mini_gray,
				scaleFactor=sF,
				minNeighbors=mN,
				minSize=(55, 55),
				flags=cv2.CASCADE_SCALE_IMAGE
			)
			if isinstance(pre_faces, tuple):
				print("no face found")
				continue
			pre_faces = list(pre_faces[0])
			pre_faces[0] += adjustments[0]
			pre_faces[1] += adjustments[1]
			faces += [pre_faces]
		bounded_box_temp = []
		for face in faces:
			sub_box = [int(face[0] - face[2]/4),
						int(face[1] - face[3]/4),
						int(face[2]*1.5),
						int(face[3]*1.5)
			]
			# ---- Out of bounds checking
			if sub_box[0] < 0:
				sub_box[0] = 0
			if sub_box[1] < 0:
				sub_box[1] = 0
			if (sub_box[0] + sub_box[2]) > gray_width:
				sub_box[2] = gray_width - sub_box[0] - 1
			if (sub_box[1] + sub_box[3]) > gray_height:
				sub_box[3] = gray_height - sub_box[1] - 1
			bounded_box_temp.append(sub_box)

		new_face += 1
	print("frame")

	# ---- Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

	# ---- Draw a rectangle around the bounded box
	for (x, y, w, h) in bounded_box:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

	if past_first_frame:
		bounded_box = bounded_box_temp

	cv2.imshow('Face Detector', frame)
	c = cv2.waitKey(7) % 0x100
	if c == 27:
		break

cap.release()
cv2.destroyAllWindows()
