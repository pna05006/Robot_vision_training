import cv2
import numpy as np
from matplotlib import pyplot as plt

## Padding function
def padding(img_raw):
    img_padding = np.insert(img_raw, 0, img_raw[0], axis=0) # 위쪽 패딩
    img_padding = np.insert(img_padding, img_padding.shape[0], img_padding[img_padding.shape[0]-1], axis=0) # 아래쪽 패딩
    img_padding = np.insert(img_padding, 0, img_padding[:,0], axis=1) # 왼쪽 패딩
    img_padding = np.insert(img_padding, img_padding.shape[1], img_padding[:,img_padding.shape[1]-1], axis=1) # 오른쪽 패딩
    return img_padding

def rm_letterbox(img_raw):
    for i in range(img_raw.shape[0]):
        if np.mean(img_raw[i]) > 1.0:
            L_box = i
            break
    return img_raw[L_box:img_raw.shape[0]-L_box]

def gray_conv(img_raw):
    # BGR 색상값
    b = img_raw[:, :, 0]
    g = img_raw[:, :, 1]
    r = img_raw[:, :, 2]
    result = ((0.299 * r) + (0.587 * g) + (0.114 * b)) # 기본 이미지가 0~1의 픽셀값으로 표현
    return result.astype(np.uint8)

## Load Image & down sampling
A1_raw = cv2.resize(cv2.imread(filename="A1.jpg", flags=cv2.IMREAD_COLOR).astype(np.uint8), dsize=(480,300)) # 16:10 해상도(480,300) (640, 400)
A2_raw = cv2.resize(cv2.imread(filename="A2.jpg", flags=cv2.IMREAD_COLOR).astype(np.uint8), dsize=(480,300))

## Remove letterbox & Padding & conversion to gray_img
A1_img = gray_conv(padding(rm_letterbox(A1_raw)))
A2_img = gray_conv(padding(rm_letterbox(A2_raw)))

## img show
cv2.imshow("A1 & A2 RAW", np.vstack((A1_img, A2_img)))
plt.imshow(np.vstack((A1_img, A2_img)), cmap='gray')
plt.show()
cv2.waitKey()
cv2.destroyAllWindows()