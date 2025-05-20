import cv2
import json
import os
import numpy as np

# 读取JSON文件
with open('pig_self_test/class_agnostic_annotations_json/new.json', 'r') as f:
    data = json.load(f)

# 构建图像ID到文件名的映射
image_id_to_info = {img['id']: img for img in data['images']}

# 创建输出目录（如果不存在）
output_dir = "json_on_img_result/class_agnostic_annotations_images"
os.makedirs(output_dir, exist_ok=True)

COLORS = {
    "bbox": (0, 255, 0),    # 绿色（BGR格式）
    "segmentation": (255, 0, 0),  # 蓝色
    "transparency": 0.3     # 分割区域填充透明度
}

# 按图像ID分组处理标注
from collections import defaultdict
annotations_by_image = defaultdict(list)
for ann in data['annotations']:
    annotations_by_image[ann['image_id']].append(ann)

# 遍历所有标注并保存结果
# for ann in data['annotations']:
for image_id, annotations in annotations_by_image.items():
    # image_id = ann['image_id']
    image_info = image_id_to_info.get(image_id)
    
    if image_info:
        # 读取原图
        basepath = 'pig_self_test/img'
        img_path = image_info['file_name']
        img_path = os.path.join(basepath, img_path)
        img = cv2.imread(img_path)

        # 获取当前标注的类别颜色
        for ann in annotations:
            if 'segmentation' in ann and ann['segmentation']:
                # 解析多边形坐标点
                for polygon in ann['segmentation']:
                    # 将坐标转换为整数类型的顶点数组
                    points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                    
                    # --- 绘制分割区域（半透明填充）---
                    # 创建一个掩码图层
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [points], COLORS['segmentation'])
                    # 将掩码与原图叠加（控制透明度）
                    cv2.addWeighted(overlay, COLORS['transparency'], img, 1 - COLORS['transparency'], 0, img)
                    
                    # --- 绘制分割轮廓（可选）---
                    cv2.polylines(img, [points], isClosed=True, color=COLORS['segmentation'], thickness=2)
            
            # 绘制边界框
            x, y, w, h = ann['bbox']
            cv2.rectangle(img, 
                        (int(x), int(y)), 
                        (int(x + w), int(y + h)), 
                        (0, 255, 0),  # 绿色框（BGR格式）
                        2  # 线宽
                        )
            
            # 添加类别标签（可选）
            category_id = ann['category_id']
            category_name = next((cat['name'] for cat in data['categories'] if cat['id'] == category_id), "Unknown")
            cv2.putText(img, 
                    f"{category_name}", 
                    (int(x), int(y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5,  # 字体大小
                    (0, 255, 0), 
                    2  # 线宽
                    )
        
        # 保存图片（保留原文件名，存放在输出目录）
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")
