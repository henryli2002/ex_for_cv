import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage.filters import convolve
from scipy.signal import convolve2d

np.set_printoptions(threshold=np.inf)


def show_np_img(image):
    pil_image = Image.fromarray(np.uint8(image))
    pil_image.show()


def hough_circle_transform(edges, radius_range, angle_step=100, merge_distance=10):
    rows, cols = edges.shape
    R_min, R_max = radius_range
    H = np.zeros((rows, cols, R_max))

    edge_points = np.argwhere(edges > 0)
    for x, y in edge_points:
        for r in range(R_min, R_max + 1):
            for t in range(0, 360, angle_step):
                a = int(x - r * np.cos(t * np.pi / 180))
                b = int(y - r * np.sin(t * np.pi / 180))
                if 0 <= a < rows and 0 <= b < cols:
                    H[a, b, r-1] += 1

    # 使用动态阈值
    mean_val = np.mean(H[H > 0])
    std_val = np.std(H[H > 0])
    threshold = mean_val + std_val

    circles = []
    for r in range(R_min, R_max + 1):
        circle_candidates = np.argwhere(H[:, :, r-1] > threshold)
        for x, y in circle_candidates:
            # 合并接近的圆心
            if not any(np.sqrt((x-x0)**2 + (y-y0)**2) < merge_distance for x0, y0, _ in circles):
                circles.append((y, x, r))
    print(circles)
    show_np_img(circles)
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