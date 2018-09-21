import cv2
import itertools
import multiprocessing
import numpy as np
import network_hands as net

cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)24/1 ! nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

cv2.startWindowThread()
cv2.namedWindow("preview")


net.initialize_flags()
estimator = net.get_estimator()

while True:
	ret, img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


	blur = cv2.GaussianBlur(gray, (5, 5), 0)
	ret, img = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	#Trazenje kontura
	im2, cont, h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	#Trazenje najvece konture
	max_area = 0
	for i in range(len(cont)):
		cnt = cont[i]
		area = cv2.contourArea(cnt)
		if(area > max_area):
			max_area = area
			ci = i
	cnt = cont[ci]
	cnt = cnt.reshape(cnt.shape[0], 2)

	#Koordinate odgovarajuceg pravouganika     
	min_y = np.amin(cnt, axis=0)[0]
	max_y = np.amax(cnt, axis=0)[0]
	min_x = np.amin(cnt, axis=0)[1]
	max_x = np.amax(cnt, axis=0)[1]

	#Odsijecanje slike
	img2 = gray[min_x:max_x+1, min_y:max_y+1]
	mask = img[min_x:max_x+1, min_y:max_y+1] / 255

	cv2.imshow('img2', np.multiply(img2, mask))

	_input = cv2.resize(img2, (28, 28))
	predictions = net.predict(estimator, _input.flatten().reshape(784, 1))
	print(predictions)

	

