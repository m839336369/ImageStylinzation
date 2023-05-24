import numpy as np
import cv2
from scipy.ndimage import convolve


def calculate_psnr(image1, image2, k1=0.05, k2=0.3, window_size=11):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel_value = np.max(image1)
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr


import os

# 指定目录路径
directory = '../results/summer2winter_yosemite/test_latest/images/'

count = 512
# 获取目录下所有图片文件
image_files = []
temp_count = count
for filename in os.listdir(directory):
    if temp_count != 0 and (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')):
        image_files.append(os.path.join(directory, filename))
        temp_count -= 1
sum = 0.0
# 每次处理两张图片为一组
for i in range(0, count, 2):
    # 读取两张图片
    file_path1 = image_files[i]
    file_path2 = image_files[i + 1] if i + 1 < len(image_files) else None

    # 使用PIL库打开图片文件
    image1 = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(file_path2, cv2.IMREAD_GRAYSCALE) if file_path2 else None

    # 在此处进行你想要的处理操作，例如显示图片、处理图像数据等
    val = calculate_psnr(image1, image2)
    sum += val
    print(val)
print(f"ssim:{sum / (count/2)}")
