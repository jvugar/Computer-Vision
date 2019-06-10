#importing packages
import numpy as np 
import argparse
import cv2

#get the input and parse the arguments

input = argparse.ArgumentParser()
input.add_argument("-i", "--image", required=True, help="input image")
input.add_argument("-p", "--protxt", required=True, help="model architecture")
input.add_argument("-w", "--weights", required=True, help="weights for the model")
input.add_argument("-c", "--confidence", type =float, default=0.5, help="threshold for detection")
arguments = vars(input.parse_args())


our_model = cv2.dnn.readNetFromCaffe(arguments["protxt"], arguments["weights"])
image = cv2.imread(arguments["image"])
resized_image = cv2.resize(image, (300,300))
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(resized_image, 1.0,
	(300, 300), (104.0, 177.0, 123.0))

#passing in the processed image to our model
our_model.setInput(blob)
detection = our_model.forward()


for i in range(0, detection.shape[2]):
	confidence = detection[0,0,i,2]

	if confidence > arguments["confidence"]:
		bounding_box = detection[0,0,i,3:7] * np.array([w,h,w,h])
		(startX, startY, endX, endY) = bounding_box.astype("int")

		#draw the bounding box
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
 
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)





















