import cv2
import copy
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
import random
from Optical_flow import *

## Make datatable
## 각 픽셀에 y, x, r value, g value, b value, v, u, class 정보가 0~1로 정규화되어 출력
## scale은 각 정보의 가중치

def get_centroids(data, K):
    centroids = np.empty((K, data.shape[2]))
    for k in range(K):
        centroids[k] = data[random.randrange(0, data.shape[0]), random.randrange(0, data.shape[1]), :]
    return centroids

def L1Norm(V1, V2):
    return np.sum(abs(V2 - V1))

def normalization(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

def min_dis_ind(data, points):
    d_min = float('inf'); min_ind = 0
    for idp, p in enumerate(points):
        d = distance.euclidean(data, p)
        if  d <= d_min:
            min_ind = idp
            d_min = d
    return min_ind

def main():
    file_set = "A"
    K = 5

    ## Load Image & down sampling & remove letterbox & padding & convert rgb image
    A1_img = bgr2rgb(padding(rm_letterbox(cv2.resize(cv2.imread(filename=file_set+"1.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(320,200))))) # 16:10 해상도(480,300) (640, 400)
    # A2_img = bgr2rgb(padding(rm_letterbox(cv2.resize(cv2.imread(filename=file_set+"2.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(320,200)))))

    data_table = np.stack([A1_img[:,:,0], A1_img[:,:,1], A1_img[:,:,2]], 2)
    cluster_table = np.zeros((data_table.shape[0], data_table.shape[1]))
    

    for x in range(data_table.shape[2]):
        data_table[:,:,x] = normalization(data_table[:,:,x])

    # centroids = np.random.rand(K, data_table.shape[2])
    # old_centroids = np.random.rand(K, data_table.shape[2])

    centroids = get_centroids(data_table, K)
    old_centroids = np.random.rand(K, data_table.shape[2])
    
    count = 0
    while True:
        #assign
        for y in range(cluster_table.shape[0]):
            for x in range(cluster_table.shape[1]):
                cluster_table[y,x] = min_dis_ind(data_table[y,x], centroids)

        #new and old assign 비교
        if np.array_equal(cluster_table, old_centroids) == True or count == 50: break 
        old_centroids = copy.deepcopy(centroids)

        #update centroid
        for k in range(K):
            u = list()
            for y in range(cluster_table.shape[0]):
                for x in range(cluster_table.shape[1]):
                    if cluster_table[y,x] == k: u.append(data_table[y,x,:])
            centroids[k] = np.mean(np.array(u), axis = 0)
            
        if np.isnan(centroids).any():
            count = 0
            centroids = get_centroids(data_table, K)
            old_centroids = np.random.rand(K, data_table.shape[2])
            print("centroids and none")
        else:
            count += 1
            print(count, "update")
    
    result_img = np.zeros(A1_img.shape)
    for k in range(K):
        u = list()
        for y in range(cluster_table.shape[0]):
            for x in range(cluster_table.shape[1]):
                if cluster_table[y,x] == k: u.append(A1_img[y,x,:].tolist())
        # print(np.mean(np.array(u), axis = 0))
        cc = np.mean(np.array(u), axis = 0)
        for y in range(cluster_table.shape[0]):
            for x in range(cluster_table.shape[1]):
                if cluster_table[y,x] == k:
                    result_img[y,x,0] = cc[0]
                    result_img[y,x,1] = cc[1]
                    result_img[y,x,2] = cc[2]
        
    plt.imshow(result_img/255.0)
    plt.show()
            
if __name__ == '__main__':
    main()