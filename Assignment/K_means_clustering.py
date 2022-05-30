import threading
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from Optical_flow import *

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
    
    # 모션 벡터로도 클러스터링 & 명도로도 클러스터링 -> 2개의 차원축으로 변화 후 세그멘테이션?
    ## display motion vector and image
    disp_result(A1_img, A1_edge_img, motion_vec_img)


if __name__ == '__main__':
    main()