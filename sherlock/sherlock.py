import cv2

from sherlock.FaceDetection import *
from sherlock.HandTracking import *

class Sherlock:
	def __init__(self, video):
		self.video = cv2.VideoCapture(video)
		self.frame = None
		self.faceDetector = FaceDetector()
		self.handTracker = HandTracker()

	def read(self):
		_, self.frame = self.video.read()
		return self.frame

	def getHand(self):
		return self.handTracker.detect(self.frame)

	def getFace(self):
		return self.faceDetector.detect(self.frame)

	def getHandVisual(self):
		return self.handTracker.visualize(self.frame.copy())

	def getFaceVisual(self):
		return self.faceDetector.visualize(self.frame.copy())

	def getFrame(self):
		return self.frame

	def release(self):
		self.video.release()
