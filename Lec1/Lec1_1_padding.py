import cv2
import numpy as np

## np_array test
# arr = np.array([[[1,1,1], [2,2,2], [3,3,3]],[[4,4,4], [5,5,5], [6,6,6]],[[7,7,7], [8,8,8], [9,9,9]]])
# arr = np.insert(arr, 0, arr[0], axis=0) # 위쪽 패딩
# arr = np.insert(arr, arr.shape[0], arr[arr.shape[0]-1], axis=0) # 아래쪽 패딩
# arr = np.insert(arr, 0, arr[:,0], axis=1) # 왼쪽 패딩
# arr = np.insert(arr, arr.shape[1], arr[:,arr.shape[1]-1], axis=1) # 오른쪽 패딩
# img_raw = arr*15/255.0
# print(arr)

## Load Image
img_raw = cv2.imread(filename="night_pic.jpg", flags=cv2.IMREAD_COLOR).astype(np.float32) / 255.0
img_ratio = img_raw.shape
img = cv2.resize(img_raw, dsize=(int(img_ratio[1]/20), int(img_ratio[0]/20))) # 패딩 잘 됬는지 보려고 한 것

## Pading Image
img_padding = np.insert(img, 0, img[0], axis=0) # 위쪽 패딩
img_padding = np.insert(img_padding, img_padding.shape[0], img_padding[img_padding.shape[0]-1], axis=0) # 아래쪽 패딩
img_padding = np.insert(img_padding, 0, img_padding[:,0], axis=1) # 왼쪽 패딩
img_padding = np.insert(img_padding, img_padding.shape[1], img_padding[:,img_padding.shape[1]-1], axis=1) # 오른쪽 패딩

## Show Image
img_padding = cv2.resize(img_padding, dsize=(img_ratio[1], img_ratio[0]))
cv2.imshow("padding", img_padding)
cv2.imshow("raw", img_raw)
cv2.waitKey()
cv2.destroyAllWindows()

def Padding_gigi(img_raw, padding_size, flags = 0):
    if flags == 0:
        img_padding = np.insert(img_raw, 0, img_raw[0], axis=0) # 위쪽 패딩
        img_padding = np.insert(img_padding, img_padding.shape[0], img_padding[img_padding.shape[0]-1], axis=0) # 아래쪽 패딩
        img_padding = np.insert(img_padding, 0, img_padding[:,0], axis=1) # 왼쪽 패딩
        img_padding = np.insert(img_padding, img_padding.shape[1], img_padding[:,img_padding.shape[1]-1], axis=1) # 오른쪽 패딩
        return img_padding
    elif flags == 1:
        return img_raw

        