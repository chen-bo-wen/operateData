import cv2
import numpy as np

# 假设这是你不规则线条的点（示例，实际你要用你的坐标）
contour_pts = np.array([[50, 150], [80, 80], [150, 120], [220, 50], [300, 150], [250, 250], [100, 250]], dtype=np.int32)

# 创建空白图或你的原始图
img = np.ones((400, 400, 3), dtype=np.uint8) * 255

# 绘制不规则线条以示意
cv2.polylines(img, [contour_pts], isClosed=False, color=(255,0,0), thickness=2)

# 计算最小外接矩形（旋转矩形）
rect = cv2.minAreaRect(contour_pts)
box = cv2.boxPoints(rect)
box = np.int0(box)

# 旋转角度为45°
rotation_angle = 45
# 计算旋转矩阵以旋转矩形
center = rect[0]
rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

# 旋转矩形的四个点
rotated_box = cv2.transform(np.array([box], dtype=np.float32), rotation_matrix)[0]
rotated_box = np.int0(rotated_box)

# 绘制矩形
cv2.polylines(img, [rotated_box], isClosed=True, color=(0,255,0), thickness=2)

# 显示结果
cv2.imshow("Rotated Bounding Box", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
