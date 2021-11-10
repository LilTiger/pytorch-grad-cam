import argparse
import os.path

import cv2
import numpy as np
import torch
import glob
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

    # 以下方法为读入单个文件夹中所有图片 若需如此 只需将下面语句开启并将100-108行注释 并将之后的代码退缩进
    # image_list = glob.glob(''.join(args.image_path))

    # 此处读入一个文件夹中所有子文件夹中的图片 并标记热力图
    # 故 --image-path 的输入路径 格式为 './insects/'
    # 特别注意文件夹名称不能为中文！
    image_list = []
    for root, dirs, files in os.walk(args.image_path):
        for d in dirs:
            images = os.listdir(root + d)
            for image in images:
                image_list.append(root + d + "/" + image)
                # 若原图为png格式 热力图为jpg格式 以下方法防止在多次运行cam.py中递归生成错误的热力图
                # 以下语句的作用是 只有当读取到的图片为 .png 格式时 才执行热力图
                if os.path.splitext(image)[1] == '.png':
                    for img in image_list:
                        rgb_img = cv2.imread(img, 1)[:, :, ::-1]
                        rgb_img = np.float32(rgb_img) / 255
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

                        # 注意一点 若原图为png 存储热力图为jpg 可以避免多次实验中 将上次实验的 热力图 也作为 输入
                        filename = os.path.basename(image)

                        # 此处split的用法为 以 . 为分隔符 取分隔后的第一个字符串
                        cam_image = cv2.resize(cam_image, (224, 224))
                        # 将生成的热力图保存到对应类文件夹中
                        cv2.imwrite(root + d + "/" + filename.split('.')[0] + '_resnet.jpg', cam_image)

                        # 默认三个模型都采用gradcam的方式
                        # 若需要指明cam具体的方法 使用以下语句
                        # 其中f'的作用是 将字符串格式化 这样就可以直接读取 变量 的值
                        # cv2.imwrite(root + d + "/" + filename.split('.')[0] + f'_{args.method}.jpg', cam_image)





        # 若需要生成 Guided Back-propagation 和 Guided Grad-Cam 将如下注释取消即可
        # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        # gb = gb_model(input_tensor, target_category=target_category)
        #
        # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(gb)

        # cv2.imwrite('./insects/' + filename.split('.')[0] + f'_{args.method}_gb.jpg', gb)
        # cv2.imwrite('./insects/' + filename.split('.')[0] + f'_{args.method}_cam_gb.jpg', cam_gb)
