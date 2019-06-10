import imutils
import cv2


img = cv2.imread("jp.png")

#extracting dimension of an image
(h, w, d) = img.shape
print("width={}, height={}, depth={}".format(w,h,d))

#extracting pixels from an image
(B, G, R) = img[300, 400]
print("R={}, G={}, B={}".format(R,G,B))

#slicing an image
roi = img[60:160, 320:420]
cv2.imshow("ROI", roi)


#resizing an image
resized = cv2.resize(img, (400, 400))

#taking aspect ration into account
r = 400.0/ w
dimension = (400, int(h*r))
resized_aspect_ratio = cv2.resize(img, dimension)
cv2.imshow("Aspect Ratio", resized_aspect_ratio)

#if you want to avoid this and just use one function:
#resized_aspect_ratio = imutils.resize(img, width=400)  #thank you to Adrian for the function; check out his awesome website: https://www.pyimagesearch.com/

#image rotation
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, -30, 1.0)
rotated_img = cv2.warpAffine(img, M, (w, h))


#function from imutils
#rotated = imutils.rotate(image, -45)

#image blurring
blur = cv2.GaussianBlur(img, (15, 15), 0)


#drawing on an image
# draw a 2px thick red rectangle surrounding the face
copy_img = img.copy()
cv2.rectangle(copy_img, (320, 60), (420, 160), (0, 0, 255), 3)

#writing on an image
copy_img_2 = img.copy()
cv2.putText(copy_img_2, "trial", (10, 25), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



#display image on screen
cv2.imshow("Fixed Resizing", resized)
cv2.imshow("ROI", roi)
cv2.imshow("Blurred", blur)
cv2.imshow("OpenCV Rotation", rotated_img)
cv2.imshow("Rectangle", copy_img)
cv2.imshow("Text", copy_img_2)
cv2.imshow("Image", img)
cv2.waitKey(0)










