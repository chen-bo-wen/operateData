import os
from PIL import Image
import numpy as np

# 输入和输出目录
image_dir = 'images/val2017'  # 原图文件夹
mask_dir = 'masks'  # 生成的 mask 文件夹
output_dir = 'overlay_images'  # 输出叠加图像的文件夹
os.makedirs(output_dir, exist_ok=True)

# 根据类别生成不一样的随机颜色
color_mapping = {
    1: (255, 0, 0),  # 类别1: 红色
    2: (0, 0, 255),  # 类别3: 蓝色
}

# 遍历 mask 文件夹中的所有 mask 文件
for mask_file in os.listdir(mask_dir):
    if mask_file.endswith('.png'):
        # 1. 加载 mask 图像
        mask_path = os.path.join(mask_dir, mask_file)
        mask_image = Image.open(mask_path).convert('L')  # 转换为灰度图像

        # 2. 获取对应的原图
        original_image_path = os.path.join(image_dir, mask_file.replace('mask_', '').replace('.png', '.jpg'))
        original_image = Image.open(original_image_path).convert('RGBA')

        # 3. 创建彩色 mask
        mask_array = np.array(mask_image)
        colored_mask = np.zeros((*mask_array.shape, 3), dtype=np.uint8)

        # 4. 基于类别为 mask 着色
        unique_classes = np.unique(mask_array)
        for category in unique_classes:
            if category == 0:  # 背景
                continue
            color = np.random.randint(0, 255, size=3)  # 随机颜色
            colored_mask[mask_array == category] = color_mapping[category]

        # 5. 创建彩色 mask 图像
        colored_mask_image = Image.fromarray(colored_mask)

        # 6. 设置透明度
        colored_mask_image.putalpha(128)  # 设置透明度

        # 7. 合并原图和彩色 mask
        combined_image = Image.alpha_composite(original_image, colored_mask_image.convert('RGBA'))

        # 8. 保存叠加后的图像
        combined_filename = os.path.join(output_dir, f'overlay_{mask_file}')
        combined_image.save(combined_filename)

        print(f'Overlay image saved: {combined_filename}')