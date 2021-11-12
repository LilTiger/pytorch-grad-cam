# 使用albumentations进行数据增强
import albumentations as A
import cv2
import os
import glob

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    # # 在保持图片大小不变的情况下随机crop
    # A.RandomSizedCrop(min_max_height=(420, 460), height=480, width=640, p=0.4),
    # 在保持图片大小不变的情况下随机平移 缩放 或旋转
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.4),
    A.OneOf([
        # always_apply与p=的作用相同 二者不必同时使用（暂定）
        A.GaussianBlur(blur_limit=(3, 7), always_apply=False, p=0.4),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.4),
             ])
])

# 注意针对不同 类文件夹 此处要做更改！
path = "./train/0001/*.jpg"

# 在类子文件夹中遍历所有 jpg 格式的image
for jpg in glob.glob(path):
    image = cv2.imread(jpg)
    # 提取路径中的图片名*.png
    jpg_file = os.path.basename(jpg)
    # 在masks文件夹中遍历所有png格式的mask

    # 去除图片名中的后缀.png 以生成*_i.png形式的image和mask
    a = str(jpg_file).split('.')[0]
    for i in range(1):
        transformed = transform(image=image)
        transformed_image = transformed['image']

        # 注意针对不同 类文件夹 此处要做更改！
        cv2.imwrite('./train/0001/' + str(a) + '_' + str(i) + '.jpg', transformed_image)

print("Augmentation finished.")
