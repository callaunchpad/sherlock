import cv2
from sherlock import Sherlock

sherlock = Sherlock(0)

while True:
	frame = sherlock.read()
	cv2.imshow('Raw Input', frame)
	cv2.imshow('Face Visualization', sherlock.getFaceVisual())
	faceData = sherlock.getFace()
	if len(faceData.faces) > 0:
		x, y, w, h = faceData.faces[0]
		face = frame[y : (y + h), x : (x + w)]
		cv2.imshow('Face', face)

	key = cv2.waitKey(30) & 0xff
	if key == 27:
		break

sherlock.release()
cv2.destroyAllWindows()
