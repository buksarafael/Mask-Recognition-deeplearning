# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
import os
from pathlib import Path

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import torch
from face_detector import FaceDetector
from model_define import DetectorTrainer

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])

face_detector = FaceDetector(prototxtPath, weightsPath)

font = cv2.FONT_HERSHEY_SIMPLEX
labels = ['Mask Off', 'Mask On']
labelColor = [(10, 0, 255), (10, 255, 0)]

# load our serialized model from disk
print("[INFO] loading model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	faces = face_detector.detect(frame)

	# loop over the detections
	for face in faces:
		# extract the confidence (i.e., probability) associated with the
		# prediction
		xStart, yStart, width, height = face
		# clamp coordinates that are outside of the image
		xStart, yStart = max(xStart, 0), max(yStart, 0)

		# predict mask label on extracted face
		faceImg = frame[yStart:yStart + height, xStart:xStart + width]
		maskNet.setInput(faceImg)
		predicted = maskNet.forward()

		# draw face frame
		cv2.rectangle(frame,
					  (xStart, yStart),
					  (xStart + width, yStart + height),
					  (126, 65, 64),
					  thickness=2)

		# center text according to the face frame
		textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
		textX = xStart + width // 2 - textSize[0] // 2

		# draw prediction label
		cv2.putText(frame,
					labels[predicted],
					(textX, yStart - 20),
					font, 1, labelColor[predicted], 2)
	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()