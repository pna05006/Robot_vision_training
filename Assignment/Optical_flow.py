import threading
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

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
    return img_gray.astype(np.float64)

## Get derivative of image
def diff_dx(img_raw):
    df_dy = np.zeros((img_raw.shape[0]-2, img_raw.shape[1]-2))
    df_dx = np.zeros((img_raw.shape[0]-2, img_raw.shape[1]-2))
    for r in range(df_dy.shape[0]):
        for c in range(df_dy.shape[1]):
            df_dy[r,c] = img_raw[r+2, c+1] - img_raw[r+0, c+1]
            df_dx[r,c] = img_raw[r+1, c+2] - img_raw[r+1, c+0]
    return df_dy, df_dx

def diff_dt(img_t, img_tf):
    df_dt = np.zeros((img_t.shape[0]-2, img_t.shape[1]-2))
    for r in range(df_dt.shape[0]):
        for c in range(df_dt.shape[1]):
            df_dt[r,c] = img_tf[r+1, c+1] - img_t[r+1, c+1]
    return df_dt

def dect_edge(img, threadinghold):
    dy, dx = diff_dx(img)
    edge_img = np.zeros(dy.shape).astype(np.uint8)
    for y in range(dy.shape[0]):
        for x in range(dy.shape[1]):
            edge_img[y, x] = 255 if np.linalg.norm([dx[y,x], dy[y,x]]) > threadinghold else 0
    return padding(edge_img)

def motion_vector(dy, dx, dt): # 3X3 patch 를 flatten 해서 list로 넘겨주기 -> v, u 의 list로 반환
    AtA = np.zeros((2,2))
    Atb = np.zeros((2,1))
    for i in range(9):
        AtA[0,0] = AtA[0,0] + dy[i]*dy[i]
        AtA[0,1] = AtA[0,1] + dy[i]*dx[i]
        AtA[1,1] = AtA[1,1] + dx[i]*dx[i]
        Atb[0,0] = Atb[0,0] - dy[i]*dt[i]
        Atb[1,0] = Atb[1,0] - dx[i]*dt[i]
    AtA[1,0] = AtA[0,1]
    if np.linalg.det(AtA) == 0:
        v = np.zeros((2,1))
    else:
        AtA_inv = np.linalg.inv(AtA)
        v = AtA_inv@Atb
    return v.flatten()

def optical_flow(A1_img, A2_img): # A1_img, A2_img 는 padding된 같은 크기의 이미지
    df_dy, df_dx = diff_dx(A1_img)
    df_dt = diff_dt(A1_img, A2_img)
    motion_vec =  np.zeros((df_dy.shape[0]-2, df_dy.shape[1]-2, 2))
    for r in range(1, motion_vec.shape[0]-1):
        for c in range(1, motion_vec.shape[1]-1):
            v =  motion_vector(df_dy[r:r+3,c:c+3].flatten(), df_dx[r:r+3,c:c+3].flatten(), df_dt[r:r+3,c:c+3].flatten())
            if np.linalg.norm(v) > 10: v = [0, 0]
            motion_vec[r,c] = v
    return padding(padding(motion_vec))

def disp_result(img, edge_img, motion_vec):
    plt.style.use('default')
    fig, ax = plt.subplots()
    for r in range(1, motion_vec.shape[0]-1):
        for c in range(1, motion_vec.shape[1]-1):
            if edge_img[r,c] > 100: # np.any(edge_img[r-1:r+2,c-1:c+2] > 100):
                color = 'black' if np.linalg.norm(motion_vec[r,c]) < 0.2 else 'deeppink'
                ax.add_patch(
                    patches.Arrow(
                        c, motion_vec.shape[0]-(r+1),
                        motion_vec[r,c,1], -motion_vec[r,c,0],
                        width=0.3,
                        edgecolor=color,
                        facecolor='white'
                    ))
    plt.xlim(-2,img.shape[1]+2)
    plt.ylim(-2,img.shape[0]+2)
    plt.imshow(np.flip(img, axis = 0), cmap='gray')
    plt.show()

def main():
    file_set = "H"

    ## Load Image & down sampling
    A1_raw = cv2.resize(cv2.imread(filename=file_set+"1.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(480,300)) # 16:10 해상도(480,300) (640, 400)
    A2_raw = cv2.resize(cv2.imread(filename=file_set+"2.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(480,300))

    ## Remove letterbox & Padding & conversion to gray_img
    A1_img = gray_conv(padding(rm_letterbox(A1_raw)))
    A2_img = gray_conv(padding(rm_letterbox(A2_raw)))

    ## Get edge image
    A1_edge_img = dect_edge(A1_img, 30)

    ## cal motion vector using optical flow algorithm
    motion_vec_img = optical_flow(A1_img, A2_img)
    
    ## display motion vector and image
    disp_result(A1_img, A1_edge_img, motion_vec_img)

if __name__ == '__main__':
    main()