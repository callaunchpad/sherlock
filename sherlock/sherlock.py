import cv2

from sherlock.FaceDetection import *
from sherlock.HandTracking import *

class Sherlock:
	"""
	Hand and Face Recognition object. Input parameter video is a int 
	representing the video feed channel.
	
	:Example:

        >>> import cv2
        >>> from sherlock import Sherlock
        >>> sherlock = Sherlock(0)
        >>> while True:
        >>> 	frame = sherlock.read()
        >>>		hand = sherlock.getHand()
        >>>		face = sherlock.getFace()
        >>>     #possibly do something with face/hand data, for now we will just show the raw input frame
        >>> 	cv2.imshow('Raw Input', frame)
        >>> key = cv2.waitKey(30) & 0xff
        >>> if key == 27:
        >>> 	break
        >>> camera.release()
        >>> cv2.destroyAllWindows()
	"""
	def __init__(self, video):
		self.video = cv2.VideoCapture(video)
		self.frame = None
		self.faceDetector = FaceDetector()
		self.handTracker = HandTracker()

	def read(self):
		"""
        Gets the next frame from the video-stream.

        :returns: the next frame
        :rtype: 3-channel matrix 
        """
		_, self.frame = self.video.read()
		return self.frame

	def getHand(self):
		"""
        Returns a hand object for the current frame containing hand
        information.

        :returns: hand object
        :rtype: Hand 
        """
		return self.handTracker.detect(self.frame)

	def getFace(self):
		"""
        Returns a face object for the current frame containing face
        information.

        :returns: face object
        :rtype: Face 
        """
		return self.faceDetector.detect(self.frame)

	def getHandVisual(self):
		return self.handTracker.visualize(self.frame.copy())

	def getFaceVisual(self):
		return self.faceDetector.visualize(self.frame.copy())

	def getFrame(self):
		"""
        Returns the current frame.

        :returns: the current frame
        :rtype: 3-channel matrix 
        """
		return self.frame

	def release(self):
		self.video.release()
