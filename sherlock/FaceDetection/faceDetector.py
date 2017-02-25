import cv2
from .face import Face

class FaceDetector:
	def __init__(self):
		self.previous = []
		self.count = 0

	def visualize(self, frame):
		"""Gives information about the bounding box as well
		as face detection."""
		obj_face = self.detect(frame)
		faces = obj_face.faces
		bounded_box = obj_face.bounded_box
		# Draw a rectangle around the faces
		for (x, y, w, h) in faces:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

		# Draw a rectangle around the bounded box
		for (x, y, w, h) in bounded_box:
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		return frame

	def detect(self, frame):
		facePath = "sherlock/data/haarcascade_frontalface_default.xml"
		"""Returns an object with information about the frame,
		bounding box, and the location of the face."""
		faceCascade = cv2.CascadeClassifier(facePath)

		# Adds 1 to number of frames since last refresh.
		self.count += 1

		sF = 1.05
		mN = 20

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		gray_width = len(gray[0])
		gray_height = len(gray)

		bounded_box = []

		# If there is no previous bounded_box, scan new frame
		if (not self.previous or self.count > 20):
			faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=sF,
			minNeighbors=mN,
			minSize=(55, 55),
			flags=cv2.CASCADE_SCALE_IMAGE
			)

			for face in faces:
				sub_box = [int(face[0] - face[2]/4),
							int(face[1] - face[3]/4),
							int(face[2]*1.5),
							int(face[3]*1.5)
				]
				# ---- Out of bounds checking
				if sub_box[0] < 0:
					sub_box[0] = 0
				if sub_box[1] < 0:
					sub_box[1] = 0
				if (sub_box[0] + sub_box[2]) > gray_width:
					sub_box[2] = gray_width - sub_box[0] - 1
				if (sub_box[1] + sub_box[3]) > gray_height:
					sub_box[3] = gray_height - sub_box[1] - 1
				bounded_box.append(sub_box)
		else:
			faces = []
			for i in self.previous:
				adjustments = [i[0], i[1]]
				mini_gray = gray[i[1]:(i[1] + i[3]), i[0]:(i[0] + i[2])]
				pre_faces = faceCascade.detectMultiScale(
					mini_gray,
					scaleFactor=sF,
					minNeighbors=mN,
					minSize=(55, 55),
					flags=cv2.CASCADE_SCALE_IMAGE
				)
				if isinstance(pre_faces, tuple):
					print("no face found")
					continue
				pre_faces = list(pre_faces[0])
				pre_faces[0] += adjustments[0]
				pre_faces[1] += adjustments[1]
				faces += [pre_faces]
			# ---- Create a bounding box for each face detected
			for face in faces:
				sub_box = [int(face[0] - face[2]/4),
							int(face[1] - face[3]/4),
							int(face[2]*1.5),
							int(face[3]*1.5)
				]
				# ---- Out of bounds checking
				if sub_box[0] < 0:
					sub_box[0] = 0
				if sub_box[1] < 0:
					sub_box[1] = 0
				if (sub_box[0] + sub_box[2]) > gray_width:
					sub_box[2] = gray_width - sub_box[0] - 1
				if (sub_box[1] + sub_box[3]) > gray_height:
					sub_box[3] = gray_height - sub_box[1] - 1
				bounded_box.append(sub_box)

		self.previous = bounded_box

		return Face(faces, bounded_box)

	def grayscale(self, frame):
		# Retrieve grayscale frame with threshold
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Apply median blur
		median_gray_frame = cv2.medianBlur(gray_frame, 5)

		# Apply global thresholds
		_, binary_filter = cv2.threshold(median_gray_frame.copy(), 127, 255, cv2.THRESH_BINARY)
		_, trunc_filter = cv2.threshold(median_gray_frame.copy(), 127, 255, cv2.THRESH_TRUNC)

		# Apply adaptive gaussian threshold
		adaptive_threshold = cv2.adaptiveThreshold(median_gray_frame.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
		adaptive_threshold = cv2.bitwise_not(adaptive_threshold)

		# Apply gaussian blur
		gaussian_gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

		# Apply Otsu threshold
		_, otsu_threshold = cv2.threshold(gaussian_gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

		return otsu_threshold
