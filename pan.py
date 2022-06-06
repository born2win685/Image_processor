import cv2
import glob
import matplotlib.pyplot as plt

imgfiles=glob.glob("img/1/*")
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
    cv2.waitKey(500)

