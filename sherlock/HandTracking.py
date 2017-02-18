import cv2

def detect(frame):
	return frame


# peter's handtracking.cpp:
# https://callaunchpad.slack.com/files/petr-lee/F47FAANA0/handtracking.cpp

CLUSTER_THRESHHOLD = 30;

""" find center of a contour.
parameters: contour - list of cv2 Points
returns: center - cv2 Point """
def findCenter(contour):
	m = cv2.moments(contour) 
	center = cv2.Point(m.m10 / m.m00, m.m01, m.m00)
	return center

""" finds the most accurate contour, (the one with the most points)
parameters: contours - list of contours (2D list of cv2 Points)
returns: contour - a contour (list of cv2 Points) that has the most points
"""
def findComplexContour(contours):
	maxPoints = 0
	contour = contours[0]
	for i in range(0, len(contours)):
		if len(contours[i]) > maxPoints:
			maxPoints = len(contours[i])
			contour = countours[i]
	return contour 

"""take a cluster of points from the convex hull near
the finger and replace it with a single point that
represents the finger for each finger
parameters: a convexHull - a contour of a convex Polygon around the body
returns: a shortened version of a convexHull that does not have a million points near each finger
""" 
def clusterConvexHull(convexHull, threshold):
	clusterHull = []
	i = 0
	while (i < len(convexHull)):
		cluster = [] #cluster of points near a finger
		hullPoint = convexHull[i]
		cluster.append(hullPoint)
		i += 1
		while (i < len(convexHull)):
			clusterPoint = convexHull[i] 
			distance = cv2.norm(hullPoint - clusterPoint)
			if distance < threshold: #testing until you move onto the next finger 
				cluster.append(clusterPoint)
				i += 1
			else:
				break
		hullPoint = cluster[len(cluster) // 2] #find the middle point in the cluster
		clusterHull.append(hullPoint) #clusterHull = cluster with only 1 point on the fingers
	return clusterHull

"""
parameters: frame - cv2 Matrix 
returns: img - cv2 Matrix
"""
def visualizeHandtracking(frame):
  	# Resize input
  	input_ = [] # cv2 Mat
    cv2.resize(frame, cv2.Size(), input_, 3, 3)  # cv::resize(normalizedDepthMap, input, cv::Size(), 3, 3, cv::INTER_CUBIC);
    
    threshhold_output = [] # cv2 2D list
    contours = [] # 2D list of cv2 Points
    hierarchy = [] # list of (lists of size 4)
    
    # Find contours
  	ret, threshhold_output = cv2.threshhold(input_, 100, 255, 0) # ???
    frame2, contours, hierarchy = cv2.findContours(threshhold_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                  
    # Find contour polygon
    contours_poly = []
    for i in range(counters.size()):
      contours_poly[i] = cv2.approxPolyDP(, 3, True)
                  
                  
	# Find largest contour
    contour = findComplexContour(contours)
    
    # Find approximated convex hull
    hull = [] # list of cv2 Points
    completeHull = [] # list of cv2 Points
    indexHull = [] # list of ints
    if len(contour) > 1:
    	cv2.convexHull(contour, completeHull, 0, 1)
        cv2.convexHull(contour, indexHull, 0, 0)
    	hull = clusterConvexHull(completeHull, CLUSTER_THRESHHOLD)
    
    # Find convexityDefects
    defects = [] # list of cv2 Vec4i
    if len(indexHull) > 3:
    	cv2.convexityDefects(contour, indexHull, defects)
    
    # Find max and min distances
    minVal, maxVal = 0, 0 # doubles
    minLoc, maxLoc = cv2.Point(), cv2.Point() # cv2 Points
    cv2.minMaxLoc(frame, minVal, maxVal, minLoc, maxLoc)
    
    # Find centor of contour
    center = findCenter(contour) # cv2 Point
    
    # Generate visual
    img = cv2.Mat.zeros(len(threshhold_output), CV_8UC3)
    color = cv2.Scalar(0, 255, 0)
    
    # Draw contours
    for i in range(0, len(contours)):
      	dummy = []
      	cv2.drawContours(img, contours_poly, color, 1, 8, dummy, 0, cv2.Point())
    
    # Draw Hull
  	index = cv2.Point()
    closest = 1 << 29
    if (len(hull) > 1):
      	for i in range(0,len(hull)):
          	p1 = hull[i] #a point object
            p2 = hull[(i + 1) % len(hull)]
            cv2.line(img, p1, p2, cv2.Scalar(255, 0, 0), 1)
            
            #Find point closest to max depth 
            if cv2.norm(p1 - maxLoc) < closets: 
              	closest = cv2.norm(p1 - maxLoc)
                index = p1
        
        cv2.line(img, center, index, cv2.Scalar(255, 0, 255), 1)
        cv2.circle(img, index, 5, cv2.Scalar(255, 0, 255) 2)


    
    # Draw defects
    endpoints = [] # list of cv2 Points
    lastStart = cv2.Point() # cv2 Point
    found = -1
    for i in range(len(defects)):
      defect = defects[i] # cv2 Vec4i
      start = contour[defect[0]] # cv2 Point
      end = contour[defect[1]] # cv2 Point
      far = contour[defect[2]] # cv2 Point
      depth = defect[3] # int; depth from edge of contour
      if cv2.norm(maxLoc - center) * 15 and cv2.pointPolygonTest(hull, far, false) > 0 and far.y < center.y:
        # draw defect
        cv2.circle(img, far, 5, cv2.Scalar(0, 255, 255), 2)
        endpoints.append(end)
        if found == -1:
          lastStart = start
          found = 0
    if found != -1:
      endpoints.append(lastStart)
    
    # Cluster fingertip locations
    endpoints = clusterConvexHull(endpoints, CLUSTER_THRESHHOLD)
  	
    # Remove endpoint closest to index finger
    closest = 1 << 29
  	closestIndex = -1
    for i in range(len(endpoints)):
      endpoint = endpoints[i] # cv2 Point
      if cv2.norm(endpoint - maxLoc) < closest:
        closest = cv2.norm(endpoint - maxLoc)
        closestIndex = i
    if closestIndex != -1:
      del endpoints[endpoints[0] + closestIndex]
    
    for endpoint in endpoints:
      cv2.circle(img, endpoint, 5, cv2.Scalar(0, 0, 255), 2)
    
    return img

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  