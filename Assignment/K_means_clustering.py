import re
import cv2
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import random
from Optical_flow import *

def get_k_point(data, K):
    k_points = np.empty((K, data.shape[2]-1))
    for k in range(K):
        k_points[k] = data[random.randrange(0, data.shape[0]), random.randrange(0, data.shape[1]), 0:3]
    return k_points

def classification(data_table, k_points, K):
    for r in range(data_table.shape[0]):
            for c in range(data_table.shape[1]):
                ud = list()
                for k in range(K):
                    ud.append(distance.euclidean(data_table[r,c,0:3], k_points[k]))
                data_table[r,c,3] = np.array(ud).argmin()
    return data_table

def update_centroids(data_table, k_points, K):
    point_dif = list()
    for k in range(K):
                u = list()
                for r in range(data_table.shape[0]):
                    for c in range(data_table.shape[1]):
                        if data_table[r,c,3] == k:
                            u.append(data_table[r,c,0:3])
                point_dif.append(np.mean(k_points[k] - np.mean(np.array(u), axis = 0)))
                k_points[k] = np.mean(u, axis = 0)
    return data_table, k_points, np.array(point_dif).max()
    
def k_mean(data_table, k_points, K):
    data_table = classification(data_table, k_points, K)
    return update_centroids(data_table, k_points, K)

def main():
    file_set = "A"
    K = 5

    ## Load Image & down sampling
    A1_raw = cv2.resize(cv2.imread(filename=file_set+"1.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(320,200)) # 16:10 해상도(480,300) (640, 400)
    A2_raw = cv2.resize(cv2.imread(filename=file_set+"2.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(320,200))

    ## Remove letterbox & Padding & conversion to gray_img
    A1_img = gray_conv(padding(rm_letterbox(A1_raw)))
    A2_img = gray_conv(padding(rm_letterbox(A2_raw)))

    ## Get edge image
    # A1_edge_img = dect_edge(A1_img, 30)

    ## cal motion vector using optical flow algorithm
    motion_vec_img = optical_flow(A1_img, A2_img)

    # disp_result(A1_img, A1_edge_img, motion_vec_img)

    data_table = np.zeros((A1_img.shape[0], A1_img.shape[1], 4)) # 각 픽셀에 x, y, value, v, u, class 정보가 담겨있음
    for r in range(A1_img.shape[0]):
        for c in range(A1_img.shape[1]):
            data_table[r, c, 0] = A1_img[r, c]/255.0
            data_table[r, c, 1] = motion_vec_img[r, c, 0]/np.max(motion_vec_img[:,:,0])
            data_table[r, c, 2] = motion_vec_img[r, c, 1]/np.max(motion_vec_img[:,:,1])
            data_table[r, c, 3] = 0
    
    k_points = get_k_point(data_table, K)
    count = 0

    while True:
        data_table, k_points, point_dif = k_mean(data_table, k_points, K)

        if point_dif < 0.00000001:
            print(count+1, "cal end")
            break
        else:
            count += 1
            print(count, "keep cal")
            print("centroids", k_points)

    k_mined_img = data_table[:,:,3]*(255/3-1)
    plt.imshow(k_mined_img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()