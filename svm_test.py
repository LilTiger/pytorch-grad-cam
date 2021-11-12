# 注意 训练/测试集不同时 对应更改directory
# 注意 insects文件夹包含原图 增强图 和 热力图 classify文件夹中只包含提取出的 特征图
import csv
import joblib
import os
import cv2
import numpy as np
import tqdm
from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score


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


# 是训练集还是测试集
directory = "./classify/train/"

# newline的作用为在writerows方法内避免空行出现
fp = open("./classify/test.csv", "w", newline='')
f_csv = csv.writer(fp)

# 创建一个列表存放 图片名称 + 预测结果
submission = []
submission.append(["id", "label"])

# 将每个类文件夹中的 图片 和 标签 创建为两个列表
feature_array, label_array = getImageData(directory)


# 创建两个列表 分别为 truth 和 predict 以绘制混淆矩阵
# 创建 truth_num 和 predict_num 存储int型的数值
truth = []
predict = []
truth_num = []
predict_num = []

# 此处为获取test文件夹中所有子文件夹中的图片名称 存放于img_list中
img_list = []
for root, dirs, files in os.walk(directory):
    for d in dirs:
        images = os.listdir(root + d)
        for image in images:
            img_list.append(root + d + "/" + image)

for x, y in tqdm.tqdm(zip(feature_array, label_array)):
    for image in img_list:
        img_temp = extractFeaturesFromImage(image)
        filename = os.path.basename(image)
        # 以下方法用来判断两矩阵是否相等
        if np.all(x == img_temp):
            x = x.reshape(1, -1)
            prediction = clf.predict(x)[0]
            # 注意 confusion_matrix 的方法中 数组可以包含字符串
            truth.append(y)
            predict.append(prediction)
            # f1 score 和 recall 方法 均需要数值
            # 以下语句将列表中的 str 类型转化为 int 类型
            truth_num = list(map(int, truth))
            predict_num = list(map(int, predict))
            print(filename.split('.')[0] + ".jpg", prediction)
            submission.append([filename.split('.')[0] + ".jpg", str(prediction)])
            # 如果定位到相同图像 则 break 跳出本层循环 不必再继续往后找 提高效率
            break

# 此处accuracy的计算方法 得出的是 所有分类 的准确率 即(预测正确的图片/所有测试集图片)
accuracy = accuracy_score(truth_num, predict_num)
# 打印混淆矩阵
print(confusion_matrix(truth, predict))

f1 = f1_score(truth_num, predict_num, average='micro')
recall = recall_score(truth_num, predict_num, average='micro')
print("accuracy: " + str(accuracy))
print("recall: " + str(recall))
print("f1 score: " + str(f1))


f_csv.writerows(submission)
fp.close()

print("Test finished.")


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

