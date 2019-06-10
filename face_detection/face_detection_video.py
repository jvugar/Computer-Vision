from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2 

# For Video Recording Purposes
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20, (640, 480))
name = 'output_final_attempt2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(name, fourcc, 20.0, (640,480))

#get the input and parse the arguments
input = argparse.ArgumentParser()
input.add_argument("-p", "--protxt", required=True, help="model architecture")
input.add_argument("-w", "--weights", required=True, help="weights for the model")
input.add_argument("-c", "--confidence", type =float, default=0.5, help="threshold for detection")
arguments = vars(input.parse_args())

our_model = cv2.dnn.readNetFromCaffe(arguments["protxt"], arguments["weights"])

start_video = VideoStream(src=0).start()
time.sleep(3.0)




#loop over the frames
while True:
	#take a frame from the video stream and resize it
	frame = start_video.read()
	frame = imutils.resize(frame,width=400)
	resized_frame = cv2.resize(frame, (300,300))
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(resized_frame, 1.0,
	(300, 300), (104.0, 177.0, 123.0))

	#passing in the processed image to our model
	our_model.setInput(blob)
	detection = our_model.forward()


	for i in range(0, detection.shape[2]):
		confidence = detection[0,0,i,2]

		#avoid weak detections by comparing to our minimum confidence
		if confidence < arguments["confidence"]:
			continue

		bounding_box = detection[0,0,i,3:7] * np.array([w,h,w,h])
		(startX, startY, endX, endY) = bounding_box.astype("int")

		#draw the bounding box
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
 
	# show the output frame
	cv2.imshow("Output", frame)
	out.write(frame)
	key = cv2.waitKey(1) & 0xFF

	#if 'w' is pressed, break from the loop
	if key == ord("w"):
		break

cv2.destroyAllWindows()
start_video.stop()
















