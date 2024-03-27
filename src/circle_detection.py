import numpy as np
from PIL import Image, ImageDraw

np.set_printoptions(threshold=np.inf)

def show_np_img(image):
    pil_image = Image.fromarray(np.uint8(image))
    pil_image.show()


def hough_circle_transform(edges, radius_range, threshold=50, accumulator_threshold=30):
    """
    Performs the Hough Circle Transform on an image represented by its edges.
    
    :param edges: NumPy array of the image edges (binary image).
    :param radius_range: Tuple (min_radius, max_radius) specifying the range of circle radii to detect.
    :param threshold: The minimum number of votes needed to consider a location as a circle center.
    :param accumulator_threshold: Threshold for the accumulator to consider a circle for drawing.
    :return: Image with detected circles and their centers marked.
    """
    # Initialize the accumulator array.
    height, width = edges.shape
    accumulator = np.zeros((height, width, radius_range[1] - radius_range[0] + 1))
    
    # Iterate over the edge pixels.
    edge_points = np.argwhere(edges)
    for x, y in edge_points:
        # Iterate over the specified range of radii.
        for radius in range(radius_range[0], radius_range[1] + 1):
            # Draw a circle in the accumulator for each edge point and possible radius.
            for angle in np.arange(0, 360):
                a = int(x - radius * np.cos(angle * np.pi / 180))
                b = int(y - radius * np.sin(angle * np.pi / 180))
                if a >= 0 and a < height and b >= 0 and b < width:
                    accumulator[a, b, radius - radius_range[0]] += 1
    
    # Identifying circles from the accumulator.
    circles = []
    for radius, acc_slice in enumerate(accumulator[:, :, :], start=radius_range[0]):
        # Find accumulator peaks above a threshold.
        acc_peaks = np.where(acc_slice > accumulator_threshold)
        for center_y, center_x in zip(*acc_peaks):
            if acc_slice[center_y, center_x] >= threshold:
                circles.append((center_x, center_y, radius))
    
    return circles

# Note: Uncomment the following line and replace `edges` and `radius_range` with actual values to use the function.
# hough_circle_transform(edges, (min_radius, max_radius))


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
    from edge_detection import edge_detection, compress_image
    image = Image.open('./data/test_images/image.png')
    # image = compress_image(image, 10)
    edges = edge_detection(image)
    result = detect_circles(edges, image, (10,50))
    result.show()