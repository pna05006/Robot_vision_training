import cv2
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
import random
from Optical_flow import *

def get_k_point(data, K):
    k_points = np.empty((K, data.shape[2]-1))
    for k in range(K):
        k_points[k] = data[random.randrange(0, data.shape[0]), random.randrange(0, data.shape[1]), 0:data.shape[2]-1]
    return k_points

def classification(data_table, k_points, K):
    for r in range(data_table.shape[0]):
        for c in range(data_table.shape[1]):
            ud = list()
            for k in range(K):
                ud.append(distance.euclidean(data_table[r,c,0:data_table.shape[2]-1], k_points[k]))
            data_table[r,c,data_table.shape[2]-1] = np.array(ud).argmin()
    return data_table

def update_centroids(data_table, k_points, K):
    point_dif = list()
    for k in range(K):
                u = list()
                for r in range(data_table.shape[0]):
                    for c in range(data_table.shape[1]):
                        if data_table[r,c,data_table.shape[2]-1] == k:
                            u.append(data_table[r,c,0:data_table.shape[2]-1])
                point_dif.append(np.mean(k_points[k] - np.mean(np.array(u), axis = 0)))
                k_points[k] = np.mean(u, axis = 0)
    return data_table, k_points, np.array(point_dif).max()
    
def k_mean(data_table, k_points, K):
    data_table = classification(data_table, k_points, K)
    return update_centroids(data_table, k_points, K)

def main():
    file_set = "K"
    K = 6
    stop_value = 0.0000001
    scale = [1,1,1,1,1,1,1] #각 픽셀에 y, x, r, g, b, v, u에 대해 k-mean 알고리즘에서 얼마나 많이 고려할지 scale 값(클수록 붆류 확실 -> 고려 많이 함)

   ## Load Image & down sampling
    A1_raw = cv2.resize(cv2.imread(filename=file_set+"1.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(320,200)) # 16:10 해상도(480,300) (640, 400)
    A2_raw = cv2.resize(cv2.imread(filename=file_set+"2.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(320,200))

    ## Remove letterbox & Padding & conversion to gray_img
    A1_img = padding(rm_letterbox(A1_raw))
    A2_img = padding(rm_letterbox(A2_raw))
    print(A1_img.shape)

    ## Get edge image
    # A1_edge_img = dect_edge(gray_conv(A1_img), 30)

    ## cal motion vector using optical flow algorithm
    motion_vec_set = np.zeros((A1_img.shape[0], A1_img.shape[1], 3 ,2))

    A1_img_rgb=np.zeros(A1_img.shape)
    A2_img_rgb=np.zeros(A2_img.shape)
    for (i1, i2) in [[0, 2], [1,1], [2,0]]:
        A1_img_rgb[:,:,i2] = A1_img[:,:,i1]
        A2_img_rgb[:,:,i2] = A2_img[:,:,i1]

    for i in range(3):
        motion_vec_set[:,:,i,:] = optical_flow(A1_img_rgb[:,:,i], A2_img_rgb[:,:,i])

    motion_vec_img = np.sum(motion_vec_set, axis=2)

    data_table = np.zeros((A1_img_rgb.shape[0], A1_img_rgb.shape[1], 8)) # 각 픽셀에 y, x, value, v, u, class 정보가 담겨있음 -> 0~1로 정규화되어있음
    for r in range(A1_img_rgb.shape[0]):
        for c in range(A1_img_rgb.shape[1]):
            data_table[r, c, 0] = r/A1_img_rgb.shape[0] * scale[0]
            data_table[r, c, 1] = c/A1_img_rgb.shape[1] * scale[1]
            data_table[r, c, 2] = A1_img_rgb[r, c, 0]/255.0 * scale[2]
            data_table[r, c, 3] = A1_img_rgb[r, c, 1]/255.0 * scale[3]
            data_table[r, c, 4] = A1_img_rgb[r, c, 2]/255.0 * scale[4]
            data_table[r, c, 5] = motion_vec_img[r, c, 0]/np.max(motion_vec_img[:,:,0]) * scale[5]
            data_table[r, c, 6] = motion_vec_img[r, c, 1]/np.max(motion_vec_img[:,:,1]) * scale[6]
            data_table[r, c, 7] = 0
    
    k_points = get_k_point(data_table, K)
    count = 0

    while True:
        data_table, k_points, point_dif = k_mean(data_table, k_points, K)

        if point_dif < stop_value or count > 20:
            print(count+1, "cal end")
            break
        else:
            count += 1
            print(count, "keep cal")
            print((point_dif-stop_value)/point_dif, point_dif, "process")
            # print("centroids", k_points)

    k_mined_img = data_table[:,:,data_table.shape[2]-1]/(K-1)*255
    plt.imshow(k_mined_img, cmap='gray')
    plt.show()

if __name__ == '__main__':
    main()