import argparse
import imutils
import cv2

#parse the arguments
input = argparse.ArgumentParser()
input.add_argument("-i", "--image", required=True, help="input image")
arguments = vars(input.parse_args())

#loading the image
img = cv2.imread(arguments["image"])

#convert to grayscale
gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#finding the edges
edges = cv2.Canny(gry, 120, 150)


#threshold; there are 2 outputs, where second is the image
threshold = cv2.threshold(gry, 225, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("threshold", threshold)


#find contours in the image
contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
img_copy = img.copy()

for i in contours:
	cv2.drawContours(img_copy, [i], -1, (240,0,159),3)
	cv2.imshow("contours", img_copy)


# draw the total number of contours found in purple
if(len(contours)) > 1:
	text = "I found {} objects!".format(len(contours))
	cv2.putText(img_copy, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
		(240, 0, 159), 2)
else:
	text = "I found 1 object!"
	cv2.putText(img_copy, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
		(240, 0, 159), 2)


cv2.imshow("Contours", img_copy)
cv2.waitKey(0)






















