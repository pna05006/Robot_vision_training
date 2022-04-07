import cv2
import numpy as np

## Load Image
# img = cv2.imread("night_pic.jpg",cv2.IMREAD_COLOR)
img_raw = cv2.imread(filename="night_pic.jpg", flags=cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

## normalized Histogram calculation
img_n_histogram = np.zeros(img_raw.shape)

for i in range(img_raw.shape[0]):
    for j in range(img_raw.shape[1]):
        img_n_histogram[i][j] = img_raw[i][j]/img_raw.size
# print(img_n_histogram.size, img_raw.size)

## variance calculation
for t in range(img_n_histogram.size):
    mu1 = 0; mu2 = 0; v0 = 0; v1 = 0
    for i in range(t):
        mu1 += img_n_histogram[i]
    for i in range(t+1, img_n_histogram.size):
        mu2 += img_n_histogram[i]



# cv2.imshow("pic", img_raw)
# cv2.waitKey()
# cv2.destroyAllWindows()