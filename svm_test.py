# 加载和调用
import csv
import joblib
import os
import cv2
import numpy as np
import glob

clf = joblib.load("./classify/svm_model.pkl")
SHAPE = (30, 30)


def extractFeaturesFromImage(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, SHAPE, interpolation=cv2.INTER_CUBIC)
    img = img.flatten()
    img = img / np.mean(img)
    return img


def getImageData(directory):
    s = 1
    feature_list = list()
    label_list = list()
    num_classes = 0
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            num_classes += 1
            images = os.listdir(root + d)
            for image in images:
                s += 1
                label_list.append(d)
                feature_list.append(extractFeaturesFromImage(root + d + "/" + image))

    return np.asarray(feature_list), np.asarray(label_list)


directory = "./classify/test/"

# newline的作用为在writerows方法内避免空行出现
fp = open("./classify/test.csv", "w", newline='')
f_csv = csv.writer(fp)

# 创建一个列表存放 图片名称 + 预测结果
submission = []
submission.append(["id", "label"])

# 将每个类文件夹中的 图片 和 标签 创建为两个列表
feature_array, label_array = getImageData(directory)


right = 0
total = 0

# 此处为获取test文件夹中所有子文件夹中的图片名称 存放于img_list中
img_list = []
for root, dirs, files in os.walk(directory):
    for d in dirs:
        images = os.listdir(root + d)
        for image in images:
            img_list.append(root + d + "/" + image)

for x, y in zip(feature_array, label_array):
    for image in img_list:
        img_temp = extractFeaturesFromImage(image)
        filename = os.path.basename(image)
        # 以下方法用来判断两矩阵是否相等
        if np.all(x == img_temp):
            x = x.reshape(1, -1)
            prediction = clf.predict(x)[0]
            print(filename.split('.')[0] + ".jpg", prediction)
            submission.append([filename.split('.')[0] + ".jpg", str(prediction)])
            if y == prediction:
                right += 1
            total += 1
            # 如果定位到相同图像 则 break 跳出本层循环 不必再继续往后找 提高效率
            break


accuracy = round(float(right) / float(total), 4)

print(str(accuracy) + "% accuracy")

f_csv.writerows(submission)
fp.close()


# img_list = glob.glob("./classify/test/*.jpg")
#
# for img in img_list:
#     img_temp = extractFeaturesFromImage(img)
#     filename = os.path.basename(img)
#     imageFeature = img_temp.reshape(1, -1)
#     print(filename.split('.')[0] + ".jpg", clf.predict(imageFeature)[0])
#     # type_insect存储为昆虫类别
#     type_insect = clf.predict(imageFeature)[0]
#     submission.append([filename.split('.')[0] + ".jpg", str(type_insect)])

print("Test finished.")
