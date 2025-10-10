import os
import cv2
import numpy as np



# 设置文件夹路径
original_folder = '/App/cbw/Datasets/TableFishFolder/TabelFishVOC/VOC2007/JPEGImages'  # 原图文件夹路径
mask_folder = '/App/cbw/TZY_AnBan_Segment/tablefish_miou_out/detection-results'     # 掩码图文件夹路径
output_folder = '/App/cbw/TZY_AnBan_Segment/tablefish_miou_out/outpu_results' # 输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义掩码颜色映射（可以根据需要调整）
category_colors = {
    0: [0, 0, 0],  # 背景颜色为黑色
    1: [255, 0, 0],  # 类别1为蓝色
    2: [0, 255, 0],  # 类别2为绿色
    3: [0, 0, 255],  # 类别3为红色
    4: [255, 255, 0],  # 类别4为黄色
    5: [255, 0, 255],  # 类别5为紫色
    6: [0, 255, 255],  # 类别6为青色
    7: [192, 192, 192],  # 类别7为灰色
    8: [128, 128, 0],       # 类别8为橄榄色 
    9: [128, 0, 128],       # 类别9为紫红色
    10: [128, 128, 128],     # 类别10为...
    # 可以继续添加更多类别颜色
}

# 处理图像
for image_filename in os.listdir(original_folder):
    # 构造原图的完整路径
    original_image_path = os.path.join(original_folder, image_filename)
    
    # 检查文件是否为图片（假设图片格式为.jpg）
    if not image_filename.lower().endswith('.jpg'):
        continue  # 跳过非图片文件
    
    # 构造对应的掩码图路径（假设掩码图与原图同名但扩展名为.png）
    mask_image_filename = os.path.splitext(image_filename)[0] + '.png'
    mask_image_path = os.path.join(mask_folder, mask_image_filename)
    
    # 检查对应的掩码图是否存在
    if not os.path.exists(mask_image_path):
        print(f"No corresponding mask found for {image_filename}, skipping...")
        continue
    
    # 读取原图和掩码图
    original_image = cv2.imread(original_image_path)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)
    
    # 调整掩码图大小以匹配原图大小（如果需要）
    if mask_image.shape[:2] != original_image.shape[:2]:
        mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # 如果掩码图不是单通道图像，转换为灰度图像
    if len(mask_image.shape) > 2:
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    # 创建一个与原图大小相同的彩色掩码图像
    colored_mask = np.zeros_like(original_image, dtype=np.uint8)
    
    # 根据掩码值应用颜色
    unique_values = np.unique(mask_image)
    for value in unique_values:
        if value in category_colors:
            colored_mask[mask_image == value] = category_colors[value]
    
    # 将彩色掩码与原图混合（这里使用加权混合，可以根据需要调整透明度）
    overlayed_image = cv2.addWeighted(original_image, 1, colored_mask, 0.5, 0)
    
    # 保存结果图像到输出文件夹
    output_image_path = os.path.join(output_folder, image_filename)
    cv2.imwrite(output_image_path, overlayed_image)
    
    print(f"Saved overlayed image for {image_filename} to {output_image_path}")