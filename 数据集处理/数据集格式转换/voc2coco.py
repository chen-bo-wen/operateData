import os
import json
import xml.etree.ElementTree as ET
import shutil

def convert_voc_to_coco_by_split(voc_root, save_root, split_name):
    """
    voc_root: VOC 数据集的根目录 (包含 Annotations, ImageSets 等)
    save_root: 目标 COCO 目录 (如 data/coco)
    split_name: 'train', 'val', 或 'test'
    """
    # 1. 路径准备
    xml_dir = os.path.join(voc_root, 'Annotations')
    img_dir = os.path.join(voc_root, 'JPEGImages')
    split_txt = os.path.join(voc_root, 'ImageSets/Main', f'{split_name}.txt')
    
    # 创建 COCO 对应的图片目录 (如 train2017)
    coco_img_dir = os.path.join(save_root, f'{split_name}2017')
    os.makedirs(coco_img_dir, exist_ok=True)
    os.makedirs(os.path.join(save_root, 'annotations'), exist_ok=True)

    # 2. 初始化 COCO 结构
    coco = {"images": [], "annotations": [], "categories": []}
    categories = {}
    bnd_id = 1

    # 3. 读取对应的 txt 文件
    with open(split_txt, 'r') as f:
        file_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"正在处理 {split_name} 分组，共 {len(file_names)} 张图片...")

    for img_id, name in enumerate(file_names):
        xml_path = os.path.join(xml_dir, f'{name}.xml')
        if not os.path.exists(xml_path):
            continue
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像信息
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        file_name = f'{name}.jpg' # 假设是 jpg，如果不是请根据实际修改
        
        # 复制图片到目标文件夹 (实现物理移动，匹配你的第二张图结构)
        src_img = os.path.join(img_dir, file_name)
        if os.path.exists(src_img):
            shutil.copy(src_img, os.path.join(coco_img_dir, file_name))

        coco["images"].append({
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": img_id
        })
        
        # 处理标注
        for obj in root.findall('object'):
            cat_name = obj.find('name').text
            if cat_name not in categories:
                cat_id = len(categories) + 1
                categories[cat_name] = cat_id
                coco["categories"].append({"id": cat_id, "name": cat_name, "supercategory": "none"})
            else:
                cat_id = categories[cat_name]
            
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)
            
            w, h = xmax - xmin, ymax - ymin
            
            coco["annotations"].append({
                "segmentation": [],
                "area": w * h,
                "iscrowd": 0,
                "image_id": img_id,
                "bbox": [xmin, ymin, w, h],
                "category_id": cat_id,
                "id": bnd_id
            })
            bnd_id += 1

    # 保存 JSON 文件
    json_path = os.path.join(save_root, 'annotations', f'instances_{split_name}2017.json')
    with open(json_path, 'w') as f:
        json.dump(coco, f, indent=4)
    print(f"{split_name} 转换完成！")

# --- 执行转换 ---
voc_root_path = '/App/lpl/Datasets/seed/VOC2007' # 你的第一张图路径
coco_save_path = '/App/lpl/Datasets/seed/coco'   # 你的第二张图目标路径

for s in ['train', 'val', 'test']:
    convert_voc_to_coco_by_split(voc_root_path, coco_save_path, s)