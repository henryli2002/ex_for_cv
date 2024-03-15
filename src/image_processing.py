from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def read_image(filepath):
    image = Image.open(filepath)
    return image
    # 实现图像读取逻辑
    pass

def convert_to_grayscale(image):
    # 转换为灰度图
    gray_image = image.convert('L')
    return gray_image
    # 实现转换为灰度图的逻辑
    pass

def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    # 填充卷积核
    for x in range(kernel_size):
        for y in range(kernel_size):
            diff = np.sqrt((x - center)**2 + (y - center)**2)
            kernel[x,y] = np.exp(-(diff**2) / (2*sigma**2))
    # 归一化卷积核
    kernel /= np.sum(kernel)

    # 将Pillow Image转换为NumPy数组
    image = convert_to_grayscale(image)  # 转换为灰度图
    image_np = np.array(image)

    # 进行卷积操作
    convolved_np = convolve2d(image_np, kernel, mode='same', boundary='wrap')

    # 将NumPy数组转换回Pillow Image
    convolved_image = Image.fromarray(np.uint8(convolved_np))

    return convolved_image
    # 实现高斯模糊滤波的逻辑
    pass

def resize_image(image, width, height):
    new_size = (width, height)  # 以像素为单位
    resized_image = image.resize(new_size)
    return resized_image
    # 实现图像大小调整的逻辑
    pass

def save_image(image, filepath):
    # 保存图像到新文件
    image.save(filepath)
    # 实现保存图像的逻辑
    pass

def show_image(image, title="Image"):
    image.show(title)
    # 实现显示图像的逻辑
    pass
