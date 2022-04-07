import cv2
import numpy

# img = cv2.imread("night_pic.jpg",cv2.IMREAD_COLOR)
img = cv2.imread(filename="night_pic.jpg", flags=cv2.IMREAD_COLOR).astype(numpy.float32) / 255.0
cv2.imshow("pic", img)
cv2.waitKey()
cv2.destroyAllWindows()