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

## Remove letterbox
def rm_letterbox(img_raw):
    for i in range(img_raw.shape[0]):
        if np.mean(img_raw[i]) > 1.0:
            L_box = i
            break
    return img_raw[L_box:img_raw.shape[0]-L_box]

## Convert BGR_img to Gray_img
def gray_conv(img_raw):
    img_gray = ((0.299 * img_raw[:, :, 2]) + (0.587 * img_raw[:, :, 1]) + (0.114 * img_raw[:, :, 0])) # cv2.imread는 bgr로 읽어옴. 각 색상의 가중치는 밝기에 영향이 큰 정도를 고려
    return img_gray.astype(np.uint8)

def derivative_img(img_raw):
    img_dev_x = np.zeros((img_raw.shape[0]-2, img_raw.shape[1]-2))
    img_dev_y = np.zeros((img_raw.shape[0]-2, img_raw.shape[1]-2))
    for r in range(img_dev_x.shape[0]):
        for c in range(img_dev_x.shape[1]):
            img_dev_x[r,c] = img_raw[r+1, c+2] - img_raw[r+1, c+0]
            img_dev_y[r,c] = img_raw[r+2, c+1] - img_raw[r+0, c+1]
    return img_dev_x, img_dev_y

def main():
    ## Load Image & down sampling
    A1_raw = cv2.resize(cv2.imread(filename="chaker.jpg", flags=cv2.IMREAD_COLOR).astype(np.uint8), dsize=(480,300)) # 16:10 해상도(480,300) (640, 400)
    # A2_raw = cv2.resize(cv2.imread(filename="A2.jpg", flags=cv2.IMREAD_COLOR).astype(np.uint8), dsize=(480,300))

    ## Remove letterbox & Padding & conversion to gray_img
    A1_img = gray_conv(padding(rm_letterbox(A1_raw)))
    # A2_img = gray_conv(padding(rm_letterbox(A2_raw)))

    ## cal derivative
    A1_dev_x, A1_dev_y = derivative_img(A1_img)
    # A2_dev_x, A2_dev_y = derivative_img(A2_img)

    ## img show
    # cv2.imshow("A1 & A2 RAW", np.vstack((A1_img, A2_img)))
    plt.imshow(A1_img, cmap='gray')
    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()

    plt.imshow(A1_dev_x, cmap='gray')
    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()