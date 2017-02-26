import cv2
from .face import Face

class FaceDetector:
	def __init__(self):
		self.previous_face_data = None
		self.refresh_rate = 0

	def visualize(self, frame):
		"""Gives information about the bounding box as well
		as face detection."""
		face_data = self.detect(frame)
		faces = face_data.faces
		local_regions = face_data.local_regions

		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

		# Draw a rectangle around the bounded box
		for (x, y, w, h) in local_regions:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		return frame

	def detect(self, frame):
		"""Returns an object with information about the frame,
		bounding box, and the location of the face."""
		face_path = "sherlock/data/haarcascade_frontalface_default.xml"
		face_cascade = cv2.CascadeClassifier(face_path)

		# Adds 1 to number of frames since last refresh.
		self.refresh_rate += 1

		# Parameters for Haar Cascade
		sF = 1.05
		mN = 3

		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray_frame_width = len(gray_frame[0])
		gray_frame_height = len(gray_frame)

		local_regions = []

		if self.previous_face_data:
			search_regions = self.previous_face_data.local_regions
			if len(search_regions) == 0:
				refresh_threshold = 5
			else:
				refresh_threshold = 20

		# Scan frame on refresh
		if (not self.previous_face_data or self.refresh_rate > refresh_threshold):
			faces = face_cascade.detectMultiScale(
				gray_frame,
				scaleFactor = sF,
				minNeighbors = mN,
				minSize = (55, 55),
				flags = cv2.CASCADE_SCALE_IMAGE
			)

			for face in faces:
				local_region = [
					int(face[0] - face[2] / 4),
					int(face[1] - face[3] / 4),
					int(face[2] * 1.5),
					int(face[3] * 1.5)
				]

				# Out of bounds checking
				if local_region[0] < 0:
					local_region[0] = 0
				if local_region[1] < 0:
					local_region[1] = 0
				if (local_region[0] + local_region[2]) > gray_frame_width:
					local_region[2] = gray_frame_width - local_region[0] - 1
				if (local_region[1] + local_region[3]) > gray_frame_height:
					local_region[3] = gray_frame_height - local_region[1] - 1
				local_regions.append(local_region)
				self.refresh_rate = 0
		else:
			faces = []
			for (x, y, w, h) in search_regions:
				gray_search_region = gray_frame[y:(y + h), x:(x + w)]
				detected_faces = face_cascade.detectMultiScale(
					gray_search_region,
					scaleFactor = sF,
					minNeighbors = mN,
					minSize = (55, 55),
					flags = cv2.CASCADE_SCALE_IMAGE
				)
				if len(detected_faces) == 0:
					print("no face found")
					continue
				face = detected_faces[0]
				face[0] += x
				face[1] += y
				faces.append(face)

			# Create a bounding box for each face detected
			for face in faces:
				local_region = [
					int(face[0] - face[2] / 4),
					int(face[1] - face[3] / 4),
					int(face[2] * 1.5),
					int(face[3] * 1.5)
				]
				# Out of bounds checking
				if local_region[0] < 0:
					local_region[0] = 0
				if local_region[1] < 0:
					local_region[1] = 0
				if (local_region[0] + local_region[2]) > gray_frame_width:
					local_region[2] = gray_frame_width - local_region[0] - 1
				if (local_region[1] + local_region[3]) > gray_frame_height:
					local_region[3] = gray_frame_height - local_region[1] - 1
				local_regions.append(local_region)

		face_data = Face(faces, local_regions)
		self.previous_face_data = face_data
		return face_data

	def processFrame(self, frame):
		# Retrieve grayscale frame with threshold
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# Apply median blur
		gray_frame = cv2.medianBlur(gray_frame, 2)
		return gray_frame
