import cv2
import glob

res = "./test_insects/00072_gradcam.jpg"
swin = "./test_insects/00072_swinT_gradcam.jpg"
vi = "./test_insects/00072_viT_gradcam.jpg"

# 使用opencv叠加图片
img1 = cv2.imread(res)
img2 = cv2.imread(swin)
img3 = cv2.imread(vi)

alpha = 0.8
meta = 0.5
gamma = 0

image_temp = cv2.addWeighted(img1, alpha, img2, meta, gamma)
image = cv2.addWeighted(image_temp, alpha, img3, meta, gamma)

cv2.imshow('image', image_temp)
cv2.waitKey(0)
