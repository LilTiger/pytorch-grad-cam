# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 08:43:23 2020
@author: Aaron
"""
import cv2
import os.path
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import tqdm

sf = []  # 空间频率
Hue = []  # 色度
Saturation = []  # 饱和度
Value = []  # 亮度
filepath = './insects/max/'  # 图像文件所在目录
pathDir = os.listdir(filepath)

for img in glob.glob(filepath + '*.jpg'):
    image = cv2.imread(img)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    v = V.ravel()[np.flatnonzero(V)]  # 亮度非零的值
    average_v = sum(v) / len(v)  # 平均亮度
    Value.append(average_v)
    s = S.ravel()[np.flatnonzero(S)]  # 饱和度非零的值
    average_s = sum(s) / len(s)  # 平均饱和度
    Saturation.append(average_s)
    h = H.ravel()[np.flatnonzero(H)]  # 色度非零的值
    average_h = sum(h) / len(h)  # 平均色度
    Hue.append(average_h)
    # plt.imshow(hsv)
    Hm_r = []  # 存储公式中的求和部分
    Hm_c = []

    M, N = 224, 224
    '''
    Rf水平方向空间频率Cf垂直方向空间频率
    M N 为图像大小
    '''
    for i in range(0, 224):
        for j in range(1, 224):
            H_r = np.square(V[i, j] - V[i, j - 1])
            Hm_r.append(H_r)

    Hm_r = sum(Hm_r)
    Rf = np.sqrt((1 / (M * N - 1)) * Hm_r) / 255  # 进行归一化处理
    for j in range(0, 224):
        for i in range(1, 224):
            H_c = np.square(V[i, j] - V[i - 1, j])
            Hm_c.append(H_c)
    Hm_c = sum(Hm_c)
    Cf = np.sqrt((1 / (M * N - 1)) * Hm_c) / 255  # 进行归一化处理
    SF = np.sqrt(Rf * Rf + Cf * Cf)  # 空间频率值
    sf.append(SF)
    '''
    保存到excel表格中
    '''

df = pd.read_excel('./insects/max/1.xlsx')  # 读取原有数据表格
# df=DataFrame(sf)#只能写入DataFrame格式文件，不能直接写列表
df['SF'] = list(sf)  # 新增一列数据注意 数据长度与原有表格行数不同会报错
# 错误提示为：
# ValueError: Length of values does not match length of index
df['Value'] = list(Value)  # 新增一列数据
df['Hue'] = list(Hue)  # 新增一列数据
df['Saturation'] = list(Saturation)  # 新增一列数据
df.to_excel('1.xls', sheet_name='Sheet1', index=False)  # 写入表格

# #  以下计算平均梯度 需要用时退注释即可
# sum = 0.0
# for img in tqdm.tqdm(glob.glob('./insects/ttt/*.jpg')):
#     img = cv2.imread(img, 0) # 后面参数为0表示取灰度图
#     sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)  # 默认ksize=3
#     sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
#     for i in range(0, 224):
#         for j in range(1, 224):
#             gm = cv2.sqrt(sobelx ** 2 + sobely ** 2)
#             avg = gm / 4
#             avgg = np.mean(avg)
#     print(avgg)
#     sum += avgg
#
# print(sum / 400)
