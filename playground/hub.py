import cv2
from sherlock import Sherlock

sherlock = Sherlock(0)

while True:
	frame = sherlock.read()
	visualFrame = sherlock.getHandVisual()

	cv2.imshow('Raw Input', frame)
	cv2.imshow('Hand Data Visualization', visualFrame)

	key = cv2.waitKey(30) & 0xff
	if key == 27:
		break

cv2.destroyAllWindows()
