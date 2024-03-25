import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d

np.set_printoptions(threshold=np.inf)


def show_np_img(image):
    pil_image = Image.fromarray(np.uint8(image))
    pil_image.show()


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

def detect_circles(edges, image, radius_range):
    """
    检测图像中的圆。
    输入:
    - image: 原始图像。
    - radius_range: 圆半径的范围。

    返回:
    - image_with_circles: 标记了检测到的圆的图像。
    """

    # 然后进行霍夫圆变换
    circles = hough_circle_transform(edges, radius_range)
    
    # 在原始图像上绘制检测到的圆
    image_with_circles = draw_detected_circles(image, circles)
    
    return image_with_circles

if __name__ == '__main__':
    from edge_detection import edge_detection
    image = Image.open('./data/test_images/image.png')
    edges = edge_detection(image)
    result = detect_circles(edges, image, (10,100))