import cv2
import numpy as np

## Load Image
# img = cv2.imread("night_pic.jpg",cv2.IMREAD_COLOR)
img_raw = cv2.imread(filename="night_pic.jpg", flags=cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

## Filter coefficients
v = np.array([[0.05],[0.25],[0.4],[0.25],[0.05]])
h = np.array([0.05,0.25,0.4,0.25,0.05])
w = h*v
# print(w)

## Burt&Adelson’ method
img_scaled = np.zeros((img_raw.shape[0] - 4, img_raw.shape[1] - 4))
print(img_raw.shape, img_scaled.shape)
x = y = list(range(-2,3)); r = 0.5
print(x,y,r)



# cv2.imshow("pic", img_raw)
# cv2.waitKey()
# cv2.destroyAllWindows()