import cv2
from sherlock import Sherlock

sherlock = Sherlock(0)

while True:
	frame = sherlock.read()
	cv2.imshow('Raw Input', frame)
	cv2.imshow('Face Visualization', sherlock.getFaceVisual())

	key = cv2.waitKey(30) & 0xff
	if key == 27:
		break

sherlock.release()
cv2.destroyAllWindows()
