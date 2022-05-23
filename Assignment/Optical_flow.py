import cv2
import numpy as np

## Padding function
def Padding(img_raw, padding_size = 1, flags = 0):
    if flags == 0 and padding_size == 1:
        img_padding = np.insert(img_raw, 0, img_raw[0], axis=0) # 위쪽 패딩
        img_padding = np.insert(img_padding, img_padding.shape[0], img_padding[img_padding.shape[0]-1], axis=0) # 아래쪽 패딩
        img_padding = np.insert(img_padding, 0, img_padding[:,0], axis=1) # 왼쪽 패딩
        img_padding = np.insert(img_padding, img_padding.shape[1], img_padding[:,img_padding.shape[1]-1], axis=1) # 오른쪽 패딩
        return img_padding
    elif flags == 1:
        return img_raw

## Load Image & down sampling
A1_raw = cv2.resize(cv2.imread(filename="A1.jpg", flags=cv2.IMREAD_COLOR).astype(np.float32) / 255.0, dsize=(640, 400)) # 16:10 해상도
A2_raw = cv2.resize(cv2.imread(filename="A2.jpg", flags=cv2.IMREAD_COLOR).astype(np.float32) / 255.0, dsize=(640, 400))

## Remove letterbox & Padding
for i in range(A1_raw.shape[0]):
    print(np.mean(A1_raw[i]))
    if np.mean(A1_raw[i]) > 0.01:
        L_box = i
        break
A1_img = Padding(A1_raw[L_box:A1_raw.shape[0]-L_box]) # 16:9의 원본 사진 비율로 잘 잘린 후 1픽셀 패딩까지 잘 됨
A2_img = Padding(A2_raw[L_box:A2_raw.shape[0]-L_box])

## img show
cv2.imshow("A1 & A2 RAW", np.vstack((A1_img, A2_img)))
cv2.waitKey()
cv2.destroyAllWindows()