import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def facedetect(file):
	haarcascades = os.path.join(os.environ["HOME"], "/.pyenv/versions/3.6.5/lib/python3.6/site-packages/cv2/data/")
	face_cascade = cv2.CascadeClassifier(os.path.join(haarcascades, "haarcascade_frontalface_default.xml"))
	img = cv2.imread(file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.11, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	plt.show()

def lookat(img, direction):
	x = direction[0]
	y = direction[1]
	cv2.rectangle(img, (0, 0), (200, 200), (255, 255, 255), -1)
	cv2.circle(img, (100, 100), 70, (157, 203, 253), -1)
	cv2.circle(img, (65, 90), 20, (0, 0, 0), -1)
	cv2.circle(img, (65, 90), 18, (255, 255, 255), -1)
	cv2.circle(img, (65 + x, 90 + y), 10, (0, 0, 0), -1)
	cv2.circle(img, (135, 90), 20, (0, 0, 0), -1)
	cv2.circle(img, (135, 90), 18, (255, 255, 255), -1)
	cv2.circle(img, (135 + x, 90 + y), 10, (0, 0, 0), -1)
	cv2.rectangle(img, (65, 134), (135, 136), (100, 100, 100), -1)

def realtimeDetect():
	# haarcascades = os.path.join(os.environ["HOME"], "/.pyenv/versions/3.6.5/lib/python3.6/site-packages/cv2/data/")
	haarcascades = "/Users/tomo/.pyenv/versions/3.6.5/lib/python3.6/site-packages/cv2/data/"
	face_cascade = cv2.CascadeClassifier(os.path.join(haarcascades, "haarcascade_frontalface_default.xml"))
	eye_cascade = cv2.CascadeClassifier(os.path.join(haarcascades, "haarcascade_eye.xml"))
	cap = cv2.VideoCapture(0)
	while True:
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.07, minNeighbors = 5)
		look = (0, 0, 0, 0)
		looking_tick = 0
		for x, y, w, h in faces:
			if look == (0, 0, 0, 0):
				look = (x, y, w, h)
				looking_tick = 0
			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
			face = img[y: y + h, x: x + w]
			face_gray = gray[y: y + h, x: x + w]
			eyes = eye_cascade.detectMultiScale(face_gray)
			for (ex, ey, ew, eh) in eyes:
				cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
				if looking_tick > 1000:
					look = (x, y, w, h)
					looking_tick = 0
		cv2.rectangle(img, (look[0], look[1]), (look[0] + look[2], look[1] + look[3]), (0, 0, 255), 2)

		direction = np.array([0, 0])
		if look != (0, 0, 0, 0):
			direction[0] = (img.shape[1] / 2) - (look[0] + (look[2] / 2))
			direction[1] = (img.shape[0] / 2) - (look[1] + (look[3] / 2))
			direction[1] = -direction[1]
			direction = direction / 30
		lookat(img, (int(direction[0]), int(direction[1])))

		cv2.imshow("video image", img)
		key = cv2.waitKey(10)
		if key == 27:  # ESCキーで終了
			break
		looking_tick += 1
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	# imgpath = input()
	# if os.path.exists(imgpath):
	# 	facedetect(imgpath)
	# else:
	# 	print(imgpath + " is not found")
	realtimeDetect()
