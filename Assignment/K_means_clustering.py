import cv2
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
import random
from Optical_flow import *

## Make datatable
## 각 픽셀에 y, x, r value, g value, b value, v, u, class 정보가 0~1로 정규화되어 출력
## scale은 각 정보의 가중치
def set_datatable(img, motion_vec_img, scale):
    data_table = np.zeros((img.shape[0], img.shape[1], 8))
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            data_table[r, c, 0] = r/img.shape[0] * scale[0]
            data_table[r, c, 1] = c/img.shape[1] * scale[1]
            data_table[r, c, 2] = img[r, c, 0]/255.0 * scale[2]
            data_table[r, c, 3] = img[r, c, 1]/255.0 * scale[3]
            data_table[r, c, 4] = img[r, c, 2]/255.0 * scale[4]
            data_table[r, c, 5] = motion_vec_img[r, c, 0]/np.max(motion_vec_img[:,:,0]) * scale[5]
            data_table[r, c, 6] = motion_vec_img[r, c, 1]/np.max(motion_vec_img[:,:,1]) * scale[6]
            data_table[r, c, 7] = 0 # class 정보
    return data_table

## Picking randomly K points by centroid from the data
def get_centroids(data, K):
    centroids = np.empty((K, data.shape[2]-1))
    for k in range(K):
        centroids[k] = data[random.randrange(0, data.shape[0]), random.randrange(0, data.shape[1]), 0:data.shape[2]-1]
    return centroids

## Assign each object to the cluster with the nearest centroid.
def classification(data_table, centroids, K):
    for r in range(data_table.shape[0]):
        for c in range(data_table.shape[1]):
            ud = list()
            #각 데이터들과 centroids사이의 euclidean거리를 구하고 가장 가까운 centroid에 따른 정보를 class에 0~K값으로 저장
            for k in range(K):
                ud.append(distance.euclidean(data_table[r,c,0:data_table.shape[2]-1], centroids[k]))
            data_table[r,c,data_table.shape[2]-1] = np.array(ud).argmin()
    # cluster된 data_table 반환
    return data_table

## Compute each centroid as the mean of the objects assigned to it.
def update_centroids(data_table, centroids, K):
    point_dif = list()
    for k in range(K): #각 cluster 마다 반복
        u = list()
        # 모든 objects 중 k cluster에 속한 데이터만 추출
        for r in range(data_table.shape[0]):
            for c in range(data_table.shape[1]):
                if data_table[r,c,data_table.shape[2]-1] == k:
                    u.append(data_table[r,c,0:data_table.shape[2]-1])
        # 직전의 centroids에서 얼마나 이동했는지 확인
        point_dif.append(np.mean(centroids[k] - np.mean(np.array(u), axis = 0)))
        # mean of the objects assigned
        centroids[k] = np.mean(np.array(u), axis = 0)
    # data_table과 업데이트 된 centroids와 가장 크게 이동했던 centroid의 이동거리 반환
    return data_table, centroids, np.array(point_dif).max()
    
## launch K-mean clustering Once
def k_mean(data_table, centroids, K):
    data_table = classification(data_table, centroids, K)
    return update_centroids(data_table, centroids, K)

def main():
    file_set = "A"
    K = 10
    stop_value = 0.000000001
    scale = [1,1,1,1,1,1,1] #각 픽셀에 y, x, r, g, b, v, u에 대해 k-mean 알고리즘에서 얼마나 많이 고려할지 scale 값(클수록 붆류 확실 -> 고려 많이 함)

    ## Load Image & down sampling
    A1_raw = cv2.resize(cv2.imread(filename=file_set+"1.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(320,200)) # 16:10 해상도(480,300) (640, 400)
    A2_raw = cv2.resize(cv2.imread(filename=file_set+"2.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(320,200))

    ## Remove letterbox & Padding
    A1_img = padding(rm_letterbox(A1_raw))
    A2_img = padding(rm_letterbox(A2_raw))

    ## transform bgr image to rgb image
    A1_img = bgr2rgb(A1_img)
    A2_img = bgr2rgb(A2_img)

    ## cal motion vector using optical flow algorithm
    motion_vec_set = np.zeros((A1_img.shape[0], A1_img.shape[1], 3 ,2))
    for i in range(3):
        motion_vec_set[:,:,i,:] = optical_flow(A1_img[:,:,i], A2_img[:,:,i])
    motion_vec_img = np.sum(motion_vec_set, axis=2)

    ## Make data table
    data_table = set_datatable(A1_img, motion_vec_img, scale)
    
    ## Get initial centroids
    centroids = get_centroids(data_table, K)
    count = 0

    ## K-mean clustering loop start
    while True:
        # K-mean clustering once
        data_table, centroids, point_dif = k_mean(data_table, centroids, K)

        # decide whether to stop
        # centroids의 변화가 일정 값 이하이거나 20번 이상 반복하면 끝
        if point_dif < stop_value or count > 50:
            print(count+1, "cal end")
            break
        else:
            count += 1
            print(count, "keep cal")
            print((point_dif-stop_value)/point_dif, point_dif, "process")

    ## display result
    ## gray image로 K개의 색으로 clustering 하여 출력
    k_mined_img = data_table[:,:,data_table.shape[2]-1]/(K-1)*255
    plt.imshow(k_mined_img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()