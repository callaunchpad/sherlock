class Hand:
	def __init__(self, contour, fingerLocations, defects, boundingBox, centroid):
		self.contour = contour
		self.fingerLocations = fingerLocations
		self.defects = defects
		self.boundingBox = boundingBox
		self.centroid = centroid
