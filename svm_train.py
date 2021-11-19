# 对于不同模型训练的svm分类器 每次可更改存储的 文件名 以供下一步操作
# 也可训练完之后更改 这样在svm_test中直接调用即可
# 注意千万不要将 不同模型 训练得出的 svm分类器 混淆
import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import tqdm
from sklearn.metrics import f1_score, confusion_matrix, recall_score, accuracy_score, precision_score
import pickle
from sklearn.ensemble import VotingClassifier

SHAPE = (56, 56)

def getImageData(directory):
    s = 1
    feature_list = list()
    label_list = list()
    num_classes = 0
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            num_classes += 1
            images = os.listdir(root + d)
            for image in tqdm.tqdm(images):
                s += 1
                label_list.append(d)
                feature_list.append(extractFeaturesFromImage(root + d + "/" + image))

    return np.asarray(feature_list), np.asarray(label_list)


def extractFeaturesFromImage(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, SHAPE, interpolation=cv2.INTER_AREA)
    img = img.flatten()
    img = img / np.mean(img)
    return img


if __name__ == "__main__":

    # 训练函数 此处固定为 训练集 路径
    directory = "./classify/tests/"

    print("extracting feathers and labels...")
    feature_array, label_array = getImageData(directory)
    print("extracting finished.")
    # 取出一部分作为验证集
    x_train, x_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.2, random_state=42)

    # 以下方法寻找最优参数 如果需要使用 将下面代码 取消注解 将79-119行 注释 即可
    # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
    tuned_parameters = [{'C': [0.5, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001, 0.0001]}]
    scores = ['precision', 'recall']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(SVC(), tuned_parameters, cv=3,
                           scoring='%s_macro' % score)

        clf.fit(x_train, y_train)
        # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
        print("Best parameters set found on development set:")
        print(clf.best_params_)

        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']

        # 看一下具体的参数间不同数值的组合后得到的分数是多少
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))

    # # 判断是否文件 而不是判断是否存在
    # # 不同模型跑完之后记得 改名字 做备份
    # if os.path.isfile("svm_model.pkl"):
    #     svm = pickle.load(open("svm_model.pkl", "rb"))
    # else:
    #     # SVC: Best parameters set found on development set:
    #     # {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}
    #     # SGDClassifier： Best parameters set found on development set:
    #     # {'loss': 'squared_hinge', 'penalty': 'elasticnet'}
    #     # svm = SVC(C=2, gamma=0.01, kernel='rbf')
    #     svm = SVC(C=2, gamma=0.0001)
    #     print("start fitting....\n")
    #     svm.fit(x_train, y_train)
    #     pickle.dump(svm, open("./classify/svm_model.pkl", "wb"))
    #
    # print("testing...\n")
    # 以下开始在验证集中测试
    # truth = []
    # predict = []
    # truth_num = []
    # predict_num = []
    #
    # for x, y in tqdm.tqdm(zip(x_test, y_test)):
    #     x = x.reshape(1, -1)
    #     prediction = svm.predict(x)[0]
    #
    #     truth.append(y)
    #     predict.append(prediction)
    #     truth_num = list(map(int, truth))
    #     predict_num = list(map(int, predict))
    #
    # accuracy = accuracy_score(truth_num, predict_num)
    # # 注意 recall_micro = f1 score_micro = accuracy
    # precision = precision_score(truth_num, predict_num, average='weighted')
    # f1 = f1_score(truth_num, predict_num, average='weighted')
    # recall = recall_score(truth_num, predict_num, average='weighted')
    # print("以下输出 在验证集中的测试结果： \n")
    # print("accuracy: " + str(accuracy))
    # print("precision: " + str(precision))
    # print("recall: " + str(recall))
    # print("f1 score: " + str(f1))

print("success")


