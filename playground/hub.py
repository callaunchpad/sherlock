import cv2
from sherlock import HandTracking, FaceDetection

camera = cv2.VideoCapture(0)

while True:
	_, frame = camera.read()
	frame = cv2.flip(frame, 1)
	cv2.imshow('Raw Input', frame)

	frame = HandTracking.detect(frame)
	cv2.imshow('Hand Tracking', frame)

	key = cv2.waitKey(30) & 0xff
	if key == 27:
		break

camera.release()
cv2.destroyAllWindows()
