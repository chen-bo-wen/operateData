import os
import numpy as np
from PIL import Image

# VOC 数据集路径
# image_dir = 'JPEGImages'
# mask_dir = 'SegmentationClass'
data_dir = os.getcwd()
# image_dir = os.path.join(data_dir, 'JPEGImages').replace('\\','/')
mask_dir = os.path.join(data_dir, 'Data/SegmentationClass').replace('\\','/') # mask_dir 数量较少

# 类别映射
class_map = {
    "background": 0,
    "egg": 1,
    "white": 2,
    "light": 3
}

# 创建掩码
for filename in os.listdir(mask_dir):
    if filename.endswith('.png'):
        # 读取分割标注
        # mask_path = os.path.join(mask_dir, filename.replace('.jpg', '.png'))
        mask_path = os.path.join(mask_dir, filename).replace('\\','/')
        mask_image = Image.open(mask_path)
        mask_array = np.array(mask_image)

        # 创建一个新的掩码
        new_mask = np.zeros_like(mask_array)

        # 将类别映射到掩码
        for class_name, class_id in class_map.items():
            new_mask[mask_array == class_id] = class_id

        # 保存或处理掩码
        new_mask_image = Image.fromarray(new_mask.astype(np.uint8))
        print(np.unique(new_mask_image))
        os.makedirs('masks', exist_ok=True)
        new_mask_image.save(f'masks/{filename.replace(".png", ".png")}')