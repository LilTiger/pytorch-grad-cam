# 完成对热力图的特征提取
# 注意 若更换训练/测试集 更改 directory_input 和 directory_output
# 注意 跑不同模型时 28行 需对应模型跑出的热力图后缀
# 注意 跑不同模型时 由于提取出的特征都会以.jpg格式命名 故不同模型 先后 提取的特征 会有覆盖
# 故 执行完extract.py 后 可以备份 对应模型的特征 或 立马利用svm分类
import cv2
import numpy as np
import glob
import os
import re
import tqdm
# 注意opencv在处理图像数据时 矩阵类型都为uint8
# 由于cv2在写入图像时将图片从0-255标准化至0-1 故乘以255即可得到灰度原图

# img_list = glob.glob('./insects/[0-9][0-9][0-9][0-9][0-9]_gradcam.jpg')

# 以下代码 将 每一类文件夹 下的所有 热力图 与 原图 按位与 以提取特征
directory_input = './insects/train/'
directory_output = './classify/train/'

# 热力图列表
heat_list = []
# 原图列表
origin_list = []

# 以下方法中 file_list 用来寻找所有图片 而 heap_list 和 origin_list 分别寻找 热力图 和 原图
for index in range(1, 21):
    if index < 10:
        file_list = glob.glob(directory_input + '000' + str(index) + '/*.jpg')
    else:
        file_list = glob.glob(directory_input + '00' + str(index) + '/*.jpg')
        # 以下寻找类文件夹中 特定模型 跑出的热力图
        # 注意跑不同模型时 下列语句 需对应更改
    for file in file_list:
        if str(file).endswith('-vitrans.jpg'):
            heat_list.append(file)
        # 若原图和热力图都为 jpg 格式 可以用正则表达式的方式匹配到原图
        # 匹配原图
        pattern = re.compile(r'[0-9]+.jpg')
        # 匹配增强图
        patterns = re.compile(r'[0-9]+_[0-9].jpg')
        if pattern.search(file) or patterns.search(file):
            origin_list.append(file)

    # 千万注意 数据量大时 小心过分嵌套 会影响效率
    for image in tqdm.tqdm(heat_list):
        img = cv2.imread(image)
        temp_name = os.path.basename(image)
        temp_name = temp_name.split('-')[0]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red = np.array([50, 0, 0])
        upper_red = np.array([255, 255, 200])
        # 提取出热力图中的显著区域
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        # 若mask中背景为白色 ground_truth为黑色 可用 255-(mask) 将颜色反转
        red_mask = 255 - red_mask

        # 此处开始将 mask 与原图按位与
        # 注意 多层循环下 效率是关键 只有在 热力图 和 原图 对应的情况下 再读入并重置图片大小即可
        for origins in origin_list:
            # 注意 获取到的basename包括后缀
            origin_name = os.path.basename(origins).split('.')[0]

            if str(temp_name) == str(origin_name):
                origin = cv2.imread(origins)
                origin = cv2.resize(origin, (224, 224))
                result = cv2.bitwise_and(origin, origin, mask=red_mask)
                if index < 10:
                    cv2.imwrite(directory_output + "/" + '000' + str(index) + "/" + origin_name.split('.')[0] +
                                '.jpg', result)
                else:
                    cv2.imwrite(directory_output + "/" + '00' + str(index) + "/" + origin_name.split('.')[0] +
                                '.jpg', result)
                break
    # 由于每次循环都读入一个文件夹中所有图片 若没有下列语句 此方法将递归将所有类子文件夹的图片append至列表中(此点与cam等方法中不同）
    # 故应该在每次循环后 清空列表 确保每次循环中的列表内容只包含该类文件夹下 所有图片
    file_list.clear()
    heat_list.clear()
    origin_list.clear()

print("extraction finished.")
