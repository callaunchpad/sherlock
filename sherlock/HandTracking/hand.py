class Hand:
	def __init__(self, contour, defects, fingerLocations, centroid, numOfHands):
		self.contour = contour
		self.fingerLocations = fingerLocations
		self.defects = defects
		self.centroid = centroid

	def __init__(self):
		self.numOfHands = 0
