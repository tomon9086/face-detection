import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

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
		eyes = None
		for x, y, w, h in faces:
			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
			face = img[y: y + h, x: x + w]
			face_gray = gray[y: y + h, x: x + w]
			eyes = eye_cascade.detectMultiScale(face_gray)
			for (ex, ey, ew, eh) in eyes:
				cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
		cv2.imshow("video image", img)
		key = cv2.waitKey(10)
		if key == 27:  # ESCキーで終了
			break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	# imgpath = input()
	# if os.path.exists(imgpath):
	# 	facedetect(imgpath)
	# else:
	# 	print(imgpath + " is not found")
	realtimeDetect()
