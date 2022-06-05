import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

## Padding function: 최외곽 픽셀 값과 동일한 값으로 padding하는 함수
def padding(img_raw):
    img_padding = np.insert(img_raw, 0, img_raw[0], axis=0) # 위쪽 패딩
    img_padding = np.insert(img_padding, img_padding.shape[0], img_padding[img_padding.shape[0]-1], axis=0) # 아래쪽 패딩
    img_padding = np.insert(img_padding, 0, img_padding[:,0], axis=1) # 왼쪽 패딩
    img_padding = np.insert(img_padding, img_padding.shape[1], img_padding[:,img_padding.shape[1]-1], axis=1) # 오른쪽 패딩
    return img_padding # padding된 이미지 반환

## Remove letterbox : 상하 래터박스를 제거해주는 함수
def rm_letterbox(img_raw):
    for i in range(img_raw.shape[0]):
        if np.mean(img_raw[i]) > 1.0: # 검은색 픽셀이 있는 행까지 카운트
            L_box = i
            break
    return img_raw[L_box:img_raw.shape[0]-L_box] # 카운트 된 행 삭제

## transform bgr image to rgb image
def bgr2rgb(bgr_img):
    rgb_img=np.zeros(bgr_img.shape)
    # bgr 순서를 rgb 순서로 변환
    for (i1, i2) in [[0, 2], [1,1], [2,0]]:
        rgb_img[:,:,i2] = bgr_img[:,:,i1]
    return rgb_img

## Convert BGR_img to Gray_img
def gray_conv(img_raw): # cv2.imread는 bgr로 읽어옴. 각 색상의 가중치는 밝기에 영향이 큰 정도를 고려
    img_gray = ((0.299 * img_raw[:, :, 2]) + (0.587 * img_raw[:, :, 1]) + (0.114 * img_raw[:, :, 0]))
    return img_gray.astype(np.float64) # gray image을 반환

## Get gradient of image
def diff_dx(img_raw): # gray이미지를 입력으로 받음(shape = (y,x,1))
    df_dy = np.zeros((img_raw.shape[0]-2, img_raw.shape[1]-2))
    df_dx = np.zeros((img_raw.shape[0]-2, img_raw.shape[1]-2))
    for r in range(df_dy.shape[0]):
        for c in range(df_dy.shape[1]): # 모든 픽셀에 대해 연산
            df_dy[r,c] = img_raw[r+2, c+1] - img_raw[r+0, c+1]
            df_dx[r,c] = img_raw[r+1, c+2] - img_raw[r+1, c+0]
    return df_dy, df_dx # y,x 방향의 gradient image 를 반환

 ## Get df/dt
def diff_dt(img_t, img_tf): # gray이미지 두장을 입력으로 받음(shape = (y,x,1))
    df_dt = np.zeros((img_t.shape[0]-2, img_t.shape[1]-2))
    for r in range(df_dt.shape[0]):
        for c in range(df_dt.shape[1]):
            df_dt[r,c] = img_tf[r+1, c+1] - img_t[r+1, c+1]
    return df_dt # df/dt image 반환

## get dege image
def dect_edge(img, Thresholdvalue):
    dy, dx = diff_dx(img) # gradient image를 받음
    edge_img = np.zeros(dy.shape).astype(np.uint8)
    for y in range(dy.shape[0]):
        for x in range(dy.shape[1]):
            # gradient의 크기가 Thresholdvalue보다 크면 value를 edge(흰색), 작으면 0(검은색)으로 바꿈
            edge_img[y, x] = 255 if np.linalg.norm([dx[y,x], dy[y,x]]) > Thresholdvalue else 0 
    return padding(edge_img) # gradient image는 이미지 사이즈가 줄었기 때문에 padding해서 반환

## compute motion vector
def motion_vector(dy, dx, dt): # Ax=b 형태의 식 풀기. 3X3 patch 를 flatten 해서 list로 받음
    AtA = np.zeros((2,2))
    Atb = np.zeros((2,1))
    for i in range(9): # 1~9까지 3*3행렬에 대해 H와 Atb 행렬 계산
        AtA[0,0] = AtA[0,0] + dy[i]*dy[i]
        AtA[0,1] = AtA[0,1] + dy[i]*dx[i]
        AtA[1,1] = AtA[1,1] + dx[i]*dx[i]
        Atb[0,0] = Atb[0,0] - dy[i]*dt[i]
        Atb[1,0] = Atb[1,0] - dx[i]*dt[i]
    AtA[1,0] = AtA[0,1]
    if np.linalg.det(AtA) == 0: # H의 역함수가 없으면 0,0으로
        v = np.zeros((2,1))
    else:
        AtA_inv = np.linalg.inv(AtA)
        v = AtA_inv@Atb # v = H^(-1)*b
    return v.flatten() # v = [v, u] 값

## launch optical flow, get motion field
def optical_flow(A1_img, A2_img): # A1_img, A2_img 는 padding된 같은 크기의 이미지
    df_dy, df_dx = diff_dx(A1_img)
    df_dt = diff_dt(A1_img, A2_img) # y, x, t 방향의 gradient 계산
    motion_vec =  np.zeros((df_dy.shape[0]-2, df_dy.shape[1]-2, 2))
    for r in range(1, motion_vec.shape[0]-1):
        for c in range(1, motion_vec.shape[1]-1): # 모든 픽셀에 대해서 motion vector 계산
            v =  motion_vector(df_dy[r:r+3,c:c+3].flatten(), df_dx[r:r+3,c:c+3].flatten(), df_dt[r:r+3,c:c+3].flatten())
            if np.linalg.norm(v) > 5.0: v = [0, 0] # motion vector의 크기 가 5.0 보다 크면 값에 문제가 있다고 판단, 0으로 초기화
            motion_vec[r,c] = v # motion field에 값 대입
    return padding(padding(motion_vec)) # 원본 이미지 크기와 같게 하기 위해 padding

## display motion field
def disp_result(img, edge_img, motion_vec): # 결과 출력
    plt.style.use('default')
    fig, ax = plt.subplots()
    for r in range(1, motion_vec.shape[0]-1):
        for c in range(1, motion_vec.shape[1]-1): # 모든 motion vector에 대해서
            if np.any(edge_img[r-1:r+2,c-1:c+2] >= 100): # edge 근처의 vector만 출력
                # vector의 크기가 1.0 보다 작으면 검정색으로, 크면 deeppink로 출력
                color = 'black' if np.linalg.norm(motion_vec[r,c]) < 1.0 else 'deeppink'
                ax.add_patch(patches.Arrow(c, motion_vec.shape[0]-(r+1), motion_vec[r,c,1], -motion_vec[r,c,0], width=0.3, edgecolor=color, facecolor='white'))
    plt.xlim(-2,img.shape[1]+2)
    plt.ylim(-2,img.shape[0]+2)
    plt.imshow(np.flip(img, axis = 0)/225)
    plt.show()

## display motion field
## disp_result와 motion field의 rgb 화살표 색 빼고 동일
def disp_result_rgb(img, edge_img, motion_vec):
    plt.style.use('default')
    fig, ax = plt.subplots()
    for a in range(3):
        # r, g, b 채널에 맞게 motion vector색 설정
        clo = 'red' if a == 0 else 'green' if a == 1 else 'blue'
        for r in range(1, motion_vec.shape[0]-1):
            for c in range(1, motion_vec.shape[1]-1):
                if np.any(edge_img[r-1:r+2,c-1:c+2] >= 100):
                    color = 'black' if np.linalg.norm(motion_vec[r,c,a]) < 0.3 else clo
                    ax.add_patch(patches.Arrow(c, motion_vec.shape[0]-(r+1), motion_vec[r,c,a,1], -motion_vec[r,c,a,0], width=0.2, edgecolor=color, facecolor='white'))
    plt.xlim(-2,img.shape[1]+2)
    plt.ylim(-2,img.shape[0]+2)
    plt.imshow(np.flip(img, axis = 0)/225)
    plt.show()

def main():
    # 0: gray이미지로 변환 후 motion vector 계산
    # 1: r,g,b 채널에서 각각 motion vector를 계산 후 sum
    mode = 1
    file_set = "A"

    ## Load Image & down sampling
    A1_raw = cv2.resize(cv2.imread(filename=file_set+"1.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(320,200)) # 16:10 해상도(480,300) (640, 400)
    A2_raw = cv2.resize(cv2.imread(filename=file_set+"2.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(320,200))

    ## Remove letterbox & Padding & conversion
    A1_img = padding(rm_letterbox(A1_raw))
    A2_img = padding(rm_letterbox(A2_raw))

    ## transform bgr image to rgb image
    A1_img = bgr2rgb(A1_img)
    A2_img = bgr2rgb(A2_img)
    
    if mode == 0: # gray이미지로 변환 후 motion vector 계산
        ## Remove letterbox & Padding & conversion to gray_img
        A1_gray = gray_conv(A1_img)
        A2_gray = gray_conv(A2_img)

        ## Get edge image
        A1_edge_img = dect_edge(A1_gray, 15)

        ## cal motion vector using optical flow algorithm
        motion_vec_img = optical_flow(A1_gray, A2_gray)
        
        ## display motion vector and image
        disp_result(A1_gray, A1_edge_img, motion_vec_img)

    else: # r,g,b 채널에서 각각 motion vector를 계산 후 sum
        ## Get edge image
        A1_edge_img = dect_edge(gray_conv(A1_img), 15)

        ## cal motion vector using optical flow algorithm
        motion_vec_set = np.zeros((A1_img.shape[0], A1_img.shape[1], 3 ,2))
        for i in range(3):
            motion_vec_set[:,:,i,:] = optical_flow(A1_img[:,:,i], A2_img[:,:,i])
        motion_vec_img = np.sum(motion_vec_set, axis=2)
        
        ## display motion vector and image
        disp_result(A1_img, A1_edge_img, motion_vec_img)
        disp_result_rgb(A1_img, A1_edge_img, motion_vec_set)

if __name__ == '__main__':
    main()