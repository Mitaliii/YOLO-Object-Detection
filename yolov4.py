# -*- coding: utf-8 -*-
"""yolov4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dO3BvswC-wd6Zl6R91QW1W2NQfExhX-J
"""

# Commented out IPython magic to ensure Python compatibility.
# pip install numpy opencv-python
import cv2, os
import matplotlib.pyplot as plt
import re
from collections import Counter
# %matplotlib inline
# !rm -fr darknet
# !git clone https://github.com/AlexeyAB/darknet/
# !sed -i 's/OPENCV=0/OPENCV=1/g' Makefile
# !sed -i 's/GPU=0/GPU=1/g' Makefile
# !sed -i 's/CUDNN=0/CUDNN=1/g' Makefile
# !sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/g' Makefile
# !apt update
# !apt-get install libopencv-dev

# cd darknet

# !wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

import numpy as np
import time
import cv2

LABELS_FILE='content//darknet//data//coco.names'
CONFIG_FILE='content//darknet//cfg//yolov4.cfg'
WEIGHTS_FILE='content//darknet//yolov4.weights'
CONFIDENCE_THRESHOLD=0.5

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

def yolo_detection(input_img):
  image = cv2.imread(input_img)
  (H, W) = image.shape[:2]

  # determine only the output layer names that we need from YOLO
  ln = net.getLayerNames()
  ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
  blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
  net.setInput(blob)
  # start = time.time()
  layerOutputs = net.forward(ln)
  # end = time.time()


  # print("[INFO] YOLO took {:.6f} seconds".format(end - start))


  # initialize our lists of detected bounding boxes, confidences, and
  # class IDs, respectively
  boxes = []
  confidences = []
  classIDs = []

  # loop over each of the layer outputs
  for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
      # extract the class ID and confidence (i.e., probability) of
      # the current object detection
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]

      # filter out weak predictions by ensuring the detected
      # probability is greater than the minimum probability
      if confidence > CONFIDENCE_THRESHOLD:
        # scale the bounding box coordinates back relative to the
        # size of the image, keeping in mind that YOLO actually
        # returns the center (x, y)-coordinates of the bounding
        # box followed by the boxes' width and height
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")

        # use the center (x, y)-coordinates to derive the top and
        # and left corner of the bounding box
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        # update our list of bounding box coordinates, confidences,
        # and class IDs
        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        classIDs.append(classID)

  # apply non-maxima suppression to suppress weak, overlapping bounding
  # boxes
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
    CONFIDENCE_THRESHOLD)
  objs_list = []
  # ensure at least one detection exists
  if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
      # extract the bounding box coordinates
      (x, y) = (boxes[i][0], boxes[i][1])
      (w, h) = (boxes[i][2], boxes[i][3])

      color = [int(c) for c in COLORS[classIDs[i]]]

      cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
      # Draw black background rectangle
      # cv2.rectangle(image, (x,x-5),(y y - 5), (0,0,0), -1)

      text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
      cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 2)
      text = "".join(re.split("[^a-zA-Z]*", text))
      objs_list.append(text)
  objs_dict = dict(Counter(objs_list)) 
  # print(objs_dict)  
  return image, objs_dict
  # show the output image
  # cv2.imwrite("output.png", image)
  # image = cv2.imread("output.png")
  # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  # plt.show()

yolo_detection('content//darknet//data//horses.jpg')

# from google.colab import files
# files.download("/content/darknet")

# !zip -r /content/darknet.zip /content/darknet
