import numpy as np
from PIL import Image, ImageDraw

np.set_printoptions(threshold=np.inf)

def show_np_img(image):
    pil_image = Image.fromarray(np.uint8(image))
    pil_image.show()


def hough_circle_transform(edges, radius_range, radius_step=1, angle_step=1, merge_distance=30):
    rows, cols = edges.shape
    R_min, R_max = radius_range
    H = np.zeros((rows, cols, R_max - R_min + 1))  # 注意调整数组大小以适应R_min到R_max的范围

    # 计算Hough累积器
    edge_points = np.argwhere(edges > 0)
    for x, y in edge_points:
        for r in range(R_min, R_max + 1, radius_step):
            for t in range(0, 360, angle_step):
                a = int(x - r * np.cos(t * np.pi / 180))
                b = int(y - r * np.sin(t * np.pi / 180))
                if 0 <= a < rows and 0 <= b < cols:
                    H[a, b, r - R_min] += 1  # 索引调整

    # 提取前10%的圆心
    H_flattened = H.flatten()
    indices_sorted = np.argsort(H_flattened)[::-1]
    top_10_percent_indices = indices_sorted[:int(len(indices_sorted) * 0.05)]
    circle_candidates = np.unravel_index(top_10_percent_indices, H.shape)
    top_10_percent_votes = H_flattened[top_10_percent_indices]  # 提取对应的投票数

    circles = []
    for i in range(len(circle_candidates[0])):
        x, y, r_idx = circle_candidates
        r = r_idx[i] + R_min  # 实际半径
        votes = top_10_percent_votes[i]
        # 检查是否与现有圆心接近
        found = False
        for j, (xc, yc, rc, vc) in enumerate(circles):
            if np.sqrt((x[i]-xc)**2 + (y[i]-yc)**2) < merge_distance:
                # 如果接近，保留投票数较多的圆心
                if votes > vc:
                    circles[j] = (x[i], y[i], r, votes)  # 用新圆心替换
                found = True
                break
        if not found:
            circles.append((x[i], y[i], r, votes))  # 添加新圆心及其投票数


    circles = [(xc, yc, rc) for xc, yc, rc, vc in circles]

    print("Detected circles:", circles)
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
    from edge_detection import edge_detection, compress_image
    image = Image.open('./data/test_images/image.png')
    # image = compress_image(image, 10)
    edges = edge_detection(image)
    result = detect_circles(edges, image, (10,50))
    result.show()