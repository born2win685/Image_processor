import cv2
import glob
import imutils
import numpy as np

imgfiles=glob.glob("img/2/*")
imgfiles.sort()
images=[]

for fname in imgfiles:
    img=cv2.imread(fname)
    images.append(img)

n_img=len(images)
print(n_img)

stitcher=cv2.Stitcher_create()
(status,result) = stitcher.stitch(images)
if status == 0:
    cv2.imshow("image",result)
    cv2.imwrite("pan_output.png",result)
    cv2.waitKey(100)

result = cv2.copyMakeBorder(result, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0,0,0))

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh_img = cv2.threshold(gray, 0, 255 , cv2.THRESH_BINARY)[1]

cv2.imshow("Threshold Image", thresh_img)
cv2.waitKey(100)

contours = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = imutils.grab_contours(contours)
areaOI = max(contours, key=cv2.contourArea)
mask = np.zeros(thresh_img.shape, dtype="uint8")
x, y, w, h = cv2.boundingRect(areaOI)
cv2.rectangle(mask, (x,y), (x + w, y + h), 255, -1)

minRectangle = mask.copy()
sub = mask.copy()

while cv2.countNonZero(sub) > 0:
    minRectangle = cv2.erode(minRectangle, None)
    sub = cv2.subtract(minRectangle, thresh_img)


contours = cv2.findContours(minRectangle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contours = imutils.grab_contours(contours)
areaOI = max(contours, key=cv2.contourArea)

cv2.imshow("minRectangle Image", minRectangle)
cv2.waitKey(100)

x, y, w, h = cv2.boundingRect(areaOI)

result = result[y:y + h, x:x + w]

cv2.imwrite("final_processed.png", result)
cv2.imshow("Stitched Image Processed", result)

cv2.waitKey(200)
