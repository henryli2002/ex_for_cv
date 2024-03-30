import numpy as np
from PIL import Image, ImageDraw

np.set_printoptions(threshold=np.inf)

def np2img(image):
    """
    把np形式的图片转化成image格式。
    
    参数:
        image: 输入图像。
    """
    pil_image = Image.fromarray(np.uint8(image))
    return pil_image


def hough_circle_transform(edges, radius_range, radius_step=1, angle_step=1, threshold=50, quantity=100, merge_distance=30):
    """
    使用霍夫圆变换在给定的边缘图像中检测圆形。
    
    参数:
        edges: 边缘检测后的图像矩阵，其中边缘应标记为255，非边缘为0或其他较低值。
        radius_range: 搜索圆形的最小和最大半径范围，格式为(R_min, R_max)。
        radius_step: 在搜索过程中半径的步长。默认为1。
        angle_step: 在计算圆形时的角度步长（以度为单位）。默认为1度。
        threshold: 识别为圆的累加器阈值。仅当圆的累加器值大于或等于此值时，才考虑为有效圆。默认为50。
        quantity: 返回的最大圆形数量。默认为100。
        merge_distance: 合并圆心很接近的圆。如果两个圆的中心距离小于此值，则仅保留累加器值较高的圆。默认为30。

    返回:
        circles: list of tuples
            检测到的圆的列表，每个元组格式为(y, x, r)，其中y,x是圆心的行列坐标，r是圆的半径。
    """
    rows, cols = edges.shape
    R_min, R_max = radius_range
    H = np.zeros((rows, cols, R_max - R_min + 1))  # 注意调整数组大小以适应R_min到R_max的范围

    # 计算Hough累积器
    strong_edge = np.argwhere(edges == 255)
    for x, y in strong_edge:
        for r in range(R_min, R_max + 1, radius_step):
            for t in range(0, 360, angle_step):
                a = int(x - r * np.cos(t * np.pi / 180))
                b = int(y - r * np.sin(t * np.pi / 180))
                if 0 <= a < rows and 0 <= b < cols:
                    H[a, b, r - R_min] += 0.5  # 索引调整

    weak_edge = np.argwhere(edges > 0)
    for x, y in weak_edge:
        for r in range(R_min, R_max + 1, radius_step):
            for t in range(0, 360, angle_step):
                a = int(x - r * np.cos(t * np.pi / 180))
                b = int(y - r * np.sin(t * np.pi / 180))
                if 0 <= a < rows and 0 <= b < cols:
                    H[a, b, r - R_min] += 0.5  # 索引调整
    


    # 提取前quantity的圆心
    H_flattened = H.flatten()
    indices_sorted = np.argsort(H_flattened)[::-1]
    top_indices = indices_sorted[:min(quantity, len(indices_sorted))]  # 提取前quantity个圆心(保证不会越界)
    circle_candidates = np.unravel_index(top_indices, H.shape)
    top_percent_votes = H_flattened[top_indices]  # 提取对应的投票数

    circles = []
    for i in range(len(circle_candidates[0])):
        x, y, r_idx = circle_candidates
        r = r_idx[i] + R_min  # 实际半径
        votes = top_percent_votes[i]
        if votes < threshold:
            break
        # 检查是否与现有圆心接近
        found = False
        for j, (yc, xc, rc, vc) in enumerate(circles):
            if np.sqrt((x[i]-xc)**2 + (y[i]-yc)**2) < merge_distance:
                # 如果接近，保留投票数较多的圆心
                if votes > vc:
                    circles[j] = (y[i], x[i], r, votes)  # 用新圆心替换
                found = True
                break
        if not found:
            circles.append((y[i], x[i], r, votes))  # 添加新圆心及其投票数


    circles = [(yc, xc, rc) for yc, xc, rc, vc in circles]

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


# for test
# if __name__ == '__main__':
#     from edge_detection import edge_detection, compress_image
#     image = Image.open('<path>').convert('RGB')
#     # image = compress_image(image, 150)
#     edges = edge_detection(image)
#     result = detect_circles(edges, image, (10,50))
#     result.show()