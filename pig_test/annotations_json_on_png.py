import cv2
import json
import os
import numpy as np

# 读取JSON文件
with open('pig_self_test/annotations_json/new.json', 'r') as f:
    data = json.load(f)

# 构建图像ID到文件名的映射
image_id_to_info = {img['id']: img for img in data['images']}

# 构建类别ID到颜色和名称的映射（注意：JSON中的颜色是RGB格式，OpenCV需要BGR）
category_id_to_color = {
    cat['id']: (
        cat['color'][2],  # B通道
        cat['color'][1],  # G通道
        cat['color'][0]   # R通道
    ) for cat in data['categories']
}
category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

# 创建输出目录
output_dir = "json_on_img_result/annotations_images"
os.makedirs(output_dir, exist_ok=True)

# 分割区域填充透明度
SEGMENTATION_ALPHA = 0.3

# 按图像ID分组处理标注
from collections import defaultdict
annotations_by_image = defaultdict(list)
for ann in data['annotations']:
    annotations_by_image[ann['image_id']].append(ann)


# 遍历所有标注
for image_id, annotations in annotations_by_image.items():
    # image_id = ann['image_id']
    image_info = image_id_to_info.get(image_id)
    
    if image_info:
        basepath = 'pig_self_test/img'
        img_path = image_info['file_name']
        img_path = os.path.join(basepath, img_path).replace("\\", "/")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: 无法读取图像 {img_path}")
            continue
        
        # 获取当前标注的类别颜色
        for ann in annotations:
            category_id = ann['category_id']
            color = category_id_to_color.get(category_id, (0, 255, 0))  # 默认绿色
            
            # 绘制分割区域（多边形）
            if 'segmentation' in ann and ann['segmentation']:
                for polygon in ann['segmentation']:
                    # 转换为OpenCV需要的顶点格式
                    points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                    
                    # 创建半透明填充层
                    overlay = img.copy()
                    cv2.fillPoly(overlay, [points], color)
                    # 叠加填充层到原图
                    cv2.addWeighted(overlay, SEGMENTATION_ALPHA, img, 1 - SEGMENTATION_ALPHA, 0, img)
                    
                    # 绘制分割轮廓
                    cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
        
            # 绘制边界框
            x, y, w, h = ann['bbox']
            cv2.rectangle(img, 
                        (int(x), int(y)), 
                        (int(x + w), int(y + h)), 
                        color, 
                        thickness=2
                        )
            
            # 添加类别标签（名称 + ID）
            label = f"{category_id_to_name.get(category_id, 'Unknown')} ({category_id})"
            cv2.putText(img, 
                    label, 
                    (int(x), int(y) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    color, 
                    thickness=2
                    )
        
        # 保存图片
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")
