import cv2

class HandTracker:
	def __init__(self):
		pass

	def detect(self, frame):
		# ### IMAGE FILTERING

		# """ find center of a contour.
		# parameters: contour - list of cv2 Points
		# returns: center - cv2 Point """
		# def findCenter(contour):
		# 	m = cv2.moments(contour) 
		# 	center = cv2.Point(m.m10 / m.m00, m.m01, m.m00)
		# 	return center

		# """ finds the most accurate contour, (the one with the most points)
		# parameters: contours - list of contours (2D list of cv2 Points)
		# returns: contour - a contour (list of cv2 Points) that has the most points
		# """
		# def findComplexContour(contours):
		# 	maxPoints = 0
		# 	contour = contours[0]
		# 	for i in range(0, len(contours)):
		# 		if len(contours[i]) > maxPoints:
		# 			maxPoints = len(contours[i])
		# 			contour = countours[i]
		# 	return contour

		# def filterHSV(frame, min_threshold, max_threshold):
		#     mask = cv2.inRange(hsv, min_threshold, max_threshold)
		#     mask = cv2.medianBlur(mask, 5)
		#     return mask

		# def largestContour(contours):
		# 	largestContour = contours[0]
		#     largestArea = cv2.contourArea(largestContour)
		#     for contour in contours:
		#       if cv2.contourArea(contour) > largestArea:
		#         largestContour = contour
		#         largestArea = cv2.contourArea(contour)
		#     return largestContour
		  
		# def findHand(frame):
		#   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		#   min_threshold = [0, 51, 40]
		#   max_threshold = [19, 174, 121]
		#   mask = filterHSV(frame, min_threshold, max_threshold)
		#   filtered, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#   handContour = largestContour(contours)
		#   return handContour
		  
		# ### HAND ANALYSIS

		# def isOpenPalm():
		#   return True

		# def analyzeOpenPalm(simpleHandContour):
		#   # Find defects and finger locations
		#   # Find center of palm
		#   defects = cv2.convexHull(simpleHandContour, returnPoints = False)
		#   fingerLocations = cv2.convexHull(simpleHandContour)
		#   center = findCenter(simpleHandContour)
		  
		#   return defects, fingerLocations, center

		# #NEED TO TEST TO FIND APPROPRIATE PERCENTAGE FOR CALCULATING EPSILON
		# def simplifyContour(complexContour):
		#   	percent = .1 # <- test values
		# 	epsilon = percent*cv2.arcLength(cnt,True)
		#     simpleContour = cv2.approxPolyDP(cnt,epsilon,True)
		#     return simpleContour
		  
		# """
		# parameters: frame - cv2 Matrix 
		# returns: img - cv2 Matrix
		# """
		# def detect(frame):
		#     handContour = findHand(frame)
		    
		return Hand()