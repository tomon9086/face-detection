import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
# %matplotlib inline

def facedetect(file):
	haarcascades = "~/.pyenv/versions/3.6.5/lib/python3.6/site-packages/cv2/data/"
	face_cascade = cv2.CascadeClassifier(os.path.join(haarcascades, "haarcascade_frontalface_default.xml"))
	img = cv2.imread(file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.11, 5)
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	plt.show()

if __name__ == "__main__":
	imgpath = input()
	if os.path.exists(imgpath):
		facedetect(imgpath)
	else:
		print(imgpath + " is not found")
