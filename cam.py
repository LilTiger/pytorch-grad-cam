# 使用方式 可以在record中找到
# 输入为文件夹名 可以循环匹配到文件夹下 所有类子文件 的所有图片 并将热力图存放至对于类子文件夹中
# 特别注意 类子文件夹名称 最好不为中文 不然需要格式转换
import argparse
import os.path
import re
import cv2
import numpy as np
import torch
import glob
import tqdm
from torchvision import models
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    model = models.resnet152(pretrained=True)

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4[-1]
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])
    target_layers = [model.layer4[-1]]

    torch.cuda.empty_cache()
    # 以下方法为读入单个文件夹中所有图片 若需如此 只需将下面语句开启并将102-112行注释 并将之后的代码退缩进
    # image_list = glob.glob(''.join(args.image_path))

    # 此处 因使用os.walk方法计算效率太低 故采用glob读入整个文件夹的图片 循环操作 类子文件夹的方式
    # 若有20个分类
    for index in range(1, 21):
        # 以下方法 匹配 0001-0020 文件夹
        if index < 10:
            image_list = glob.glob(''.join(args.image_path) + '000' + str(index) + '/*.jpg')
        else:
            image_list = glob.glob(''.join(args.image_path) + '00' + str(index) + '/*.jpg')
        # 注意 此处若多缩进一处 会导致 index 从 10 开始 试着推演或者debug一下
        for img in tqdm.tqdm(image_list):
            # 以下语句的作用是 只有当读取到的图片为 原图 或 增强 格式时 才执行热力图 避免递归执行热力图 以及执行其它模型的热力图
            # 匹配增强图
            pattern = re.compile(r'[0-9]+_[0-9].jpg')
            # 匹配原图
            patterns = re.compile(r'[0-9]+.jpg')
            if pattern.search(img) or patterns.search(img):
                rgb_img = cv2.imread(img, 1)[:, :, ::-1]
                rgb_img = np.float32(rgb_img) / 255
                rgb_img = cv2.resize(rgb_img, (224, 224))
                # 因模型的pretrained为true,预训练数据集来自于ImageNet，以下的mean与std为ImageNet标准化的参数
                input_tensor = preprocess_image(rgb_img,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

                # If None, returns the map for the highest scoring category.
                # Otherwise, targets the requested category.
                target_category = None

                # Using the with statement ensures the context is freed, and you can
                # recreate different CAM objects in a loop.
                cam_algorithm = methods[args.method]
                with cam_algorithm(model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda) as cam:

                    # AblationCAM and ScoreCAM have batched implementations.
                    # You can override the internal batch size for faster computation.
                    cam.batch_size = 32

                    grayscale_cam = cam(input_tensor=input_tensor,
                                        target_category=target_category,
                                        aug_smooth=args.aug_smooth,
                                        eigen_smooth=args.eigen_smooth)

                    # Here grayscale_cam has only one image in the batch
                    grayscale_cam = grayscale_cam[0, :]

                    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                filename = os.path.basename(img)

                # 将生成的热力图保存到对应类文件夹中
                if index < 10:
                    cv2.imwrite(''.join(args.image_path) + '000' + str(index) + '/' + filename.split('.')[0] + '-resnet.jpg',
                                cam_image)
                else:
                    cv2.imwrite(''.join(args.image_path) + '00' + str(index) + '/' + filename.split('.')[0] + '-resnet.jpg',
                                cam_image)
        if index < 10:
            print("generating resnet grad-cam for class 000" + str(index) + " finished!")
        else:
            print("generating resnet grad-cam for class 00" + str(index) + " finished!")

    # 默认三个模型都采用gradcam的方式
    # 若需要指明cam具体的方法 使用以下语句
    # 其中f'的作用是 将字符串格式化 这样就可以直接读取 变量 的值
    # cv2.imwrite(root + d + "/" + filename.split('.')[0] + f'_{args.method}.jpg', cam_image)

