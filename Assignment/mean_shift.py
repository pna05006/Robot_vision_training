import cv2
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from Optical_flow import *

def classification(data_table, k_points, R):
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
    R = 0.01
    stop_value = 0.000000001
    scale = [0.7,0.7,1,1,1,0.2,0.2] #각 픽셀에 x, y, r, g, b, v, u에 대해 k-mean 알고리즘에서 얼마나 많이 고려할지 scale 값

   ## Load Image & down sampling
    A1_raw = cv2.resize(cv2.imread(filename=file_set+"1.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(80,50)) # 16:10 해상도(480,300) (640, 400)
    A2_raw = cv2.resize(cv2.imread(filename=file_set+"2.jpg", flags=cv2.IMREAD_COLOR).astype(np.float64), dsize=(80,50))

    ## Remove letterbox & Padding & conversion to gray_img
    A1_img = padding(rm_letterbox(A1_raw))
    A2_img = padding(rm_letterbox(A2_raw))

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

    data_table = np.zeros((A1_img_rgb.shape[0], A1_img_rgb.shape[1], 8)) # 각 픽셀에 x, y, value, v, u, class 정보가 담겨있음 -> 0~1로 정규화되어있음
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
    
    mean_shift_info = np.zeros((data_table.shape[0], data_table.shape[1], 2))
    mean_shift_info = mean_shift_info[:,:,0:data_table.shape[2]-1] = data_table[:,:,0:data_table.shape[2]-1]
    count = 0
    for r in range(mean_shift_info.shape[0]):
        for c in range(mean_shift_info.shape[1]):
            count += 1
            # 각 점에 대해서 최종값에 도달할 때 까지 각각 반복
            point_dif = 100
            while True:
                u = list()
                for x in range(data_table.shape[0]):
                    for y in range(data_table.shape[1]):
                        if distance.euclidean(mean_shift_info[r,c,0:data_table.shape[2]-1], data_table[x,y,0:data_table.shape[2]-1]) < R:
                            u.append(data_table[x,y,0:data_table.shape[2]-1])
                print(len(u))
                point_dif = (np.mean(mean_shift_info[r,c] - np.mean(np.array(u), axis = 0)))
                mean_shift_info[r,c] = np.mean(np.array(u), axis=0)
                
                if np.array(point_dif).max() < stop_value:
                    print(count/(mean_shift_info.shape[0]*mean_shift_info.shape[1]), "% pixel pass")
                    break
    print("end")

    # K = 0

    # k_mined_img = data_table[:,:,data_table.shape[2]-1]/(K-1)*255
    # plt.imshow(k_mined_img, cmap='gray')
    # plt.show()

if __name__ == '__main__':
    main()