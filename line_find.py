import cv2
import numpy as np

screen_height = 480
screen_width = 640


# 从路径读取图片
def image_read(path):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (screen_width, screen_height))
    # cv2.imshow("s", img)
    return img


# 创建一个白色图片
def create_wait():
    img = np.ones((screen_height, screen_width), dtype=np.uint8)  # random.random()方法后面不能加数据类型
    # img = np.random.random((3,3)) #生成随机数都是小数无法转化颜色,无法调用cv2.cvtColor函数
    img[:, :] = 225
    return img


def calculate_line(dst2):
    """线性回归计算"""
    points = []
    sum_x = 0
    sum_y = 0
    for y in range(screen_height):
        for x in range(screen_width):
            if dst2[y][x] == 0:
                points.append([x, y])
    for point in points:
        sum_x += point[0]
        sum_y += point[1]
    points_len = len(points)
    average_x = sum_x / points_len
    average_y = sum_y / points_len

    sum_up = 0
    sum_down = 0
    for point in points:
        sum_up += point[0] * point[1]
        sum_down += point[0] ** 2
    sum_up -= points_len * average_x * average_y
    sum_down -= points_len * average_x ** 2
    b = sum_up / sum_down
    a = average_y - b * average_x
    end = {'b': b, 'a': a}
    return end


def line_error(path='image/test_1.jpg'):
    """读取图片"""
    frame = image_read(path)
    """灰度化"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    """二值化"""
    threshold_image = cv2.threshold(gray, 60, 225,
                                    cv2.THRESH_BINARY)
    dst = threshold_image[1]
    """取反"""
    dst = cv2.subtract(create_wait(), dst)
    """腐蚀"""
    kernel = np.ones((5, 5), np.uint8)
    dst = cv2.erode(dst, kernel, iterations=1)
    """显示处理效果"""
    # cv2.imshow("g0", dst)
    """寻找轮廓"""
    contours = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    """画出轮廓,填充"""
    """cv2.drawContours(frame, contour, -3, (0, 255, 0), -3)"""
    """选区最大的轮廓"""
    area = []
    for k in range(len(contour)):
        area.append(cv2.contourArea(contour[k]))
    max_idx = np.argmax(np.array(area))
    """画出最大轮廓"""
    dst2 = cv2.drawContours(create_wait(), contour, max_idx, (0, 0, 225), -1)
    return calculate_line(dst2)


def show_result(path='image/test_1.jpg'):
    img = image_read(path)
    end = line_error(path)
    print(end)
    for i in range(screen_width):
        # 图像　圆心坐标　半径　颜色　边框宽度　圆边框线型　圆心坐标和半径小数点位数
        cv2.circle(
            img,
            (int(i), int(end['b'] * i + end['a'])),
            2,
            (0, 0, 225),
            0,
            1)
    """显示轮廓图片"""
    cv2.imshow(path[6:], img)


show_result('image/test_1.jpg')
show_result('image/test_2.jpg')
show_result('image/test_3.jpg')
cv2.waitKey(0)
