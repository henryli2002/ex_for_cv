import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d


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

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

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

            except IndexError as e:
                pass
    
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

def edge_detection(image, low_threshold=0.05, high_threshold=0.15):
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
    gradient_magnitude, gradient_direction = compute_gradients(convolved_image)
    nms_image = non_maximum_suppression(gradient_magnitude, gradient_direction)
    edges = double_threshold(nms_image, low_threshold, high_threshold)
    
    return edges


def draw_detected_circles(image, circles, display_centers=True):
    """
    在图像上绘制检测到的圆及其圆心。
    
    参数:
        image_path: 输入图像的路径。
        circles: 检测到的圆的列表，每个元素为(x, y, radius)。
        display_centers: 是否显示圆心，默认为True。
    """
    # 创建一个可以用来对图像进行绘制的对象
    draw = ImageDraw.Draw(image)
    
    for circle in circles:
        # Pillow中的绘制方法需要左上角和右下角的坐标
        left_up = (circle[0]-circle[2], circle[1]-circle[2])
        right_down = (circle[0]+circle[2], circle[1]+circle[2])
        
        # 绘制圆，Pillow绘制圆需要用ellipse方法，通过给定外接矩形来绘制
        draw.ellipse([left_up, right_down], outline="green", width=2)
        
        # 绘制圆心
        if display_centers:
            center = (circle[0], circle[1])
            draw.ellipse([center[0]-2, center[1]-2, center[0]+2, center[1]+2], fill="red")
    
    return image

def hough_circle_transform(edges, radius_range):
    rows, cols = edges.shape
    R_max = np.max(radius_range)
    # 创建一个三维数组：两个维度对应图像空间，第三个维度对应半径的可能值
    H = np.zeros((rows, cols, R_max))
    
    # 对每个边缘点进行处理
    edge_points = np.argwhere(edges > 0)
    for x, y in edge_points:
        for r in range(radius_range[0], radius_range[1] + 1):
            # 为每个可能的半径，在预期的圆心位置增加投票
            for t in range(360):
                a = int(x - r * np.cos(t * np.pi / 180))
                b = int(y - r * np.sin(t * np.pi / 180))
                if a >= 0 and a < rows and b >= 0 and b < cols:
                    H[a, b, r-1] += 1
    
    # 寻找局部最大值作为圆心
    threshold = np.max(H) * 0.5
    circles = []
    for r in range(radius_range[0], radius_range[1] + 1):
        # 获取半径为r时的圆心位置
        circle_candidates = np.argwhere(H[:, :, r-1] > threshold)
        for x, y in circle_candidates:
            circles.append((y, x, r))
    
    return circles

def detect_circles(image, radius_range):
    """
    检测图像中的圆。
    输入:
    - image: 原始图像。
    - radius_range: 圆半径的范围。

    返回:
    - image_with_circles: 标记了检测到的圆的图像。
    """
    # 首先进行边缘检测
    edges = edge_detection(image)
    
    # 然后进行霍夫圆变换
    circles = hough_circle_transform(edges, radius_range)
    
    # 在原始图像上绘制检测到的圆
    image_with_circles = draw_detected_circles(image, circles)
    
    return image_with_circles


