import argparse
import cv2
import numpy as np
import torch
import timm
import glob
import os
import re
import tqdm

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
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

    parser.add_argument(
        '--method',
        type=str,
        default='scorecam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python swin_trans.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

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

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
    model.eval()

    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.layers[-1].blocks[-1].norm2]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)

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
                rgb_img = cv2.resize(rgb_img, (224, 224))
                rgb_img = np.float32(rgb_img) / 255
                input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])

                # If None, returns the map for the highest scoring category.
                # Otherwise, targets the requested category.
                target_category = None

                # AblationCAM and ScoreCAM have batched implementations.
                # You can override the internal batch size for faster computation.
                cam.batch_size = 32

                grayscale_cam = cam(input_tensor=input_tensor,
                                    target_category=target_category,
                                    eigen_smooth=args.eigen_smooth,
                                    aug_smooth=args.aug_smooth)

                # Here grayscale_cam has only one image in the batch
                grayscale_cam = grayscale_cam[0, :]

                filename = os.path.basename(img)
                cam_image = show_cam_on_image(rgb_img, grayscale_cam)

                if index < 10:
                    cv2.imwrite(''.join(args.image_path) + '000' + str(index) + '/' + filename.split('.')[0] + '-swintrans.jpg',
                                cam_image)
                else:
                    cv2.imwrite(''.join(args.image_path) + '00' + str(index) + '/' + filename.split('.')[0] + '-swintrans.jpg',
                                cam_image)
        if index < 10:
            print("generating swin transformer grad-cam for class 000" + str(index) + " finished!")
        else:
            print("generating swin transformer grad-cam for class 00" + str(index) + " finished!")