import numpy as np
import cv2

from sherlock.HandTracking import Hand

class HandTracker:
	def __init__(self):
		pass

	### IMAGE FILTERING

	def findCentroid(self, contour):
		m = cv2.moments(contour)
		centroid = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
		return centroid

	def findComplexContour(self, contours):
		maxPoints = 0
		contour = contours[0]
		for i in range(0, len(contours)):
			if len(contours[i]) > maxPoints:
				maxPoints = len(contours[i])
				contour = countours[i]
		return contour

	def filterImage(self, frame, min_threshold, max_threshold):
		mask = cv2.inRange(frame, min_threshold, max_threshold)
		mask = cv2.medianBlur(mask, 5)
		return mask

	def largestContour(self, contours):
		largestContour = contours[0]
		largestArea = cv2.contourArea(largestContour)
		for contour in contours:
			if cv2.contourArea(contour) > largestArea:
				largestContour = contour
				largestArea = cv2.contourArea(contour)
		return largestContour

	def findHand(self, frame):
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		min_threshold = np.array([0, 35, 105])
		max_threshold = np.array([20, 116, 237])
		mask = self.filterImage(frame, min_threshold, max_threshold)
		filtered, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		if len(contours) == 0:
			return False
		handContour = self.largestContour(contours)
		return handContour

	### HAND ANALYSIS

	def isOpenPalm(self):
		return True

	def analyzeOpenPalm(self, handContour):
		simpleContour = handContour # self.simplifyContour(handContour)
		defects = cv2.convexHull(simpleContour, returnPoints = False)
		fingerLocations = cv2.convexHull(simpleContour)
		centroid = self.findCentroid(simpleContour)
		return defects, fingerLocations, centroid

	# NEED TO TEST TO FIND APPROPRIATE PERCENTAGE FOR CALCULATING EPSILON
	def simplifyContour(self, complexContour):
		percent = .1 # <- test values
		epsilon = percent * cv2.arcLength(complexContour, True)
		simpleContour = cv2.approxPolyDP(complexContour, epsilon, True)
		return simpleContour

	def detect(self, frame):
		contour = self.findHand(frame)
		if contour == False:
			return Hand()
		defects, fingerLocations, centroid = self.analyzeOpenPalm(contour)
		return Hand(contour, defects, fingerLocations, centroid)

	def visualize(self, frame):
		hand = self.detect(frame)
		if hand.numOfHands == 0:
			return frame
		cv2.drawContours(frame, [hand.contour], -1, (0, 255, 0), 3)
		return frame
