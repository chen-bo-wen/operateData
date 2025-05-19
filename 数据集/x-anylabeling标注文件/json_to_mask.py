import json
import numpy as np
from PIL import Image
import os
from shapely.geometry import Point, Polygon

# 得到的是 灰度图像（没有什么颜色）

# 输入和输出目录
input_dir = 'jsons/val2017'
output_dir = 'masks'
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有 JSON 文件
for json_file in os.listdir(input_dir):
    if json_file.endswith('.json'):
        json_path = os.path.join(input_dir, json_file)

        # 1. 加载 JSON 文件
        with open(json_path, 'r') as f:
            data = json.load(f)

        # 2. 提取图像信息
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        shapes = data['shapes']

        # 3. 定义类别列表
        categories = sorted(set(shape['label'] for shape in shapes))
        category_to_index = {category: idx + 1 for idx, category in enumerate(categories)}  # 从 1 开始

        # 4. 创建一个空白的 mask
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # 5. 处理每个标注
        for shape in shapes:
            if shape['shape_type'] == 'polygon':  # 确保是多边形
                points = shape['points']
                polygon = Polygon(points)

                # 获取多边形的类别索引
                category_index = category_to_index[shape['label']]

                # 获取多边形的边界框
                min_x, min_y, max_x, max_y = polygon.bounds

                # 遍历多边形内的所有点
                for y in range(int(min_y), int(max_y) + 1):
                    for x in range(int(min_x), int(max_x) + 1):
                        if polygon.contains(Point(x, y)):
                            mask[y, x] = category_index  # 将该点设置为对应类别的索引

        # 6. 保存 mask 图像
        mask_image = Image.fromarray(mask)  # mask 中的值从 0 到 n
        mask_image = mask_image.convert('L')  # 转换为灰度图像
        mask_filename = os.path.join(output_dir, f'mask_{json_file[:-5]}.png')  # 为每个 mask 文件命名
        mask_image.save(mask_filename)

        print(f'Mask for {json_file} has been saved to the directory: {output_dir}')