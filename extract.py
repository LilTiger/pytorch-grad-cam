# 完成对热力图的特征提取
import cv2
import numpy as np
import glob
import os
import time
# 注意opencv在处理图像数据时 矩阵类型都为uint8
# 由于cv2在写入图像时将图片从0-255标准化至0-1 故乘以255即可得到灰度原图

# img_list = glob.glob('./insects/[0-9][0-9][0-9][0-9][0-9]_gradcam.jpg')

# 以下代码 将 每一类文件夹 下的所有 热力图 与 原图 按位与 以提取特征
directory_input = './insects/train/'
directory_output = './classify/train/'

# 热力图列表
heap_list = []
# 原图列表
origin_list = []

for root, dirs, files in os.walk(directory_input):
    for d in dirs:
        images = os.listdir(root + d)
        for file in images:
            # 以下寻找类文件夹中 特定模型 跑出的热力图
            # 注意跑不同模型时 下列语句 需对应更改
            if str(file).endswith('_resnet.jpg'):
                heap_list.append(root + d + "/" + file)
            if str(file).endswith('.png'):
                origin_list.append(root + d + "/" + file)
            for image in heap_list:
                img = cv2.imread(image)
                temp_name = os.path.basename(image)
                temp_name = temp_name.split('_')[0]
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower_red = np.array([50, 0, 0])
                upper_red = np.array([150, 255, 255])
                # 提取出热力图中的显著区域
                red_mask = cv2.inRange(hsv, lower_red, upper_red)

                # 若mask中背景为白色 ground_truth为黑色 可用 255-(mask) 将颜色反转
                red_mask = 255 - red_mask

                # 此处开始与将mask与原图按位与
                for origins in origin_list:
                    origin = cv2.imread(origins)
                    origin = cv2.resize(origin, (224, 224))
                    # 注意 获取到的basename包括后缀
                    origin_names = os.path.basename(origins)
                    origin_name = origin_names.split('.')[0]
                    if str(temp_name) == str(origin_name):

                        result = cv2.bitwise_and(origin, origin, mask=red_mask)

                        # 写入输出路径中 对应类子文件夹
                        for root2, dir2, file2 in os.walk(directory_output):
                            for dir_out in dir2:
                                # place代表该 图像 来自于 哪个类
                                # 只有当 图像 来自的类 和 要输出的类 对应相等时 才存入特征
                                place = str(origins).split('/')[3]
                                if place == str(dir_out):
                                    cv2.imwrite(directory_output + dir_out + "/" + origin_name.split('.')[0] + '.jpg',
                                                result)
                                    # 写入之后不必再遍历文件夹 跳出本层循环
                                    break
                            break

print("extraction finished.")
