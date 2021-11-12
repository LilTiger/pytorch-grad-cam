# 以下函数完成 jpg 到 png 格式的 双向转换 并删除 原 jpg/png 格式文件
# 注意 此函数 *默认* 只针对单个文件夹下所有图片的操作
# 注意 此函数 *默认* 为 jpg 转为 png 并删除 .png 格式文件~
# 如需相反操作 注意 更改file.replace 以及 name.endswith
import os
import cv2

# 若输入输出为同一文件夹 那么下面两个路径的参数设为相同即可
input_path = './insects/train/'
output_path = './insects/train/'


def transform(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        for name in files:
            file = os.path.join(root, name)
            # print('transform' + name)
            im = cv2.imread(file)
            # 学习以下语句的用法
            cv2.imwrite(os.path.join(output_path, name.replace('jpg', 'png')), im)
            if name.endswith(".jpg"):
                os.remove(os.path.join(root, name))


if __name__ == '__main__':

    print("Start to transform!")
    transform(input_path, output_path)
    print("Transform end!")

