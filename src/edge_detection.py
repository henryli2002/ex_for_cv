import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d

np.set_printoptions(threshold=np.inf)


def show_np_img(image):
    pil_image = Image.fromarray(np.uint8(image))
    pil_image.show()

def convert_to_grayscale(image):
    # 转换为灰度图
    gray_image = image.convert('L')
    return gray_image

def apply_gaussian_blur(image, kernel_size=9, sigma=2.0):
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

def compute_gradients(image):
    """
    计算图像的梯度幅值和方向。
    输入:
    - image: 灰度图像的NumPy数组。

    返回:
    - gradient_magnitude: 梯度幅值的NumPy数组。
    - gradient_direction: 梯度方向的NumPy数组。
    """
    # Sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # 应用Sobel算子
    gx = convolve(image, sobel_x)
    gy = convolve(image, sobel_y)

    # 计算梯度幅值和方向
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    gradient_direction = np.arctan2(gy, gx) * (180 / np.pi)  # 转换为角度
    return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    """
    非极大值抑制。
    输入:
    - gradient_magnitude: 梯度幅值的NumPy数组。
    - gradient_direction: 梯度方向的NumPy数组。

    返回:
    - nms_image: 非极大值抑制后的图像。
    """
    M, N = gradient_magnitude.shape
    Z = np.zeros((M,N), dtype=np.float32)
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180
    q = 0
    r = 0
    for i in range(1, M-1):
        for j in range(1, N-1):
            #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = gradient_magnitude[i, j+1]
                r = gradient_magnitude[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = gradient_magnitude[i+1, j-1]
                r = gradient_magnitude[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = gradient_magnitude[i+1, j]
                r = gradient_magnitude[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = gradient_magnitude[i-1, j-1]
                r = gradient_magnitude[i+1, j+1]

            if (gradient_magnitude[i,j] >= q) and (gradient_magnitude[i,j] >= r):
                Z[i,j] = gradient_magnitude[i,j]
            else:
                Z[i,j] = 0
    print(Z)
    return Z

def double_threshold(nms_image, low_threshold, high_threshold):
    """
    双阈值处理。
    输入:
    - nms_image: 非极大值抑制后的图像。
    - low_threshold: 低阈值。
    - high_threshold: 高阈值。

    返回:
    - result_image: 双阈值处理后的图像。
    """
    strong_edge = nms_image >= high_threshold
    weak_edge = (nms_image >= low_threshold) & (nms_image < high_threshold)
    
    # 将结果标记为强边缘(255)，弱边缘(75)，非边缘(0)
    result = np.zeros_like(nms_image)
    result[strong_edge] = 255
    result[weak_edge] = 75
    return result

def edge_detection(image, low_threshold=5, high_threshold=15):
    """
    边缘检测主函数。
    输入:
    - image: 灰度图像的NumPy数组。
    - low_threshold: 低阈值比例。
    - high_threshold: 高阈值比例。

    返回:
    - edges: 检测到的边缘的二值图像。
    """
    convolved_image = apply_gaussian_blur(image)
    convolved_image = apply_gaussian_blur(convolved_image)
    gradient_magnitude, gradient_direction = compute_gradients(convolved_image)
    nms_image = non_maximum_suppression(gradient_magnitude, gradient_direction)
    edges = double_threshold(nms_image, low_threshold, high_threshold)
    show_np_img(edges)
    return edges


if __name__ == '__main__':
    image = Image.open('./data/test_images/image.png')
    result = edge_detection(image)
