import cv2
import numpy as np

# 这个处理的效果也没有很好。。
# 橙子的处理参考的是下面的代码，但是好像无法正常运行
# https://blog.csdn.net/jacke121/article/details/95736092?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-95736092-blog-141146905.235^v43^pc_blog_bottom_relevance_base2&spm=1001.2101.3001.4242.1&utm_relevant_index=3
def remove_reflection_v2(img):
    # 1. 生成高光掩码（阈值稍低）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, mask = cv2.threshold(hsv[:, :, 2], 235, 255, cv2.THRESH_BINARY)

    # 2. 形态学开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 3. 只膨胀一次（防止侵占边缘）
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 4. 为边缘提供参考像素
    pad = 2
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                 borderType=cv2.BORDER_REPLICATE)
    mask_pad = cv2.copyMakeBorder(mask, pad, pad, pad, pad,
                                  borderType=cv2.BORDER_CONSTANT, value=0)

    # 5. inpaint（使用 TEALE）
    inpainted = cv2.inpaint(img_pad, mask_pad, inpaintRadius=5,
                            flags=cv2.INPAINT_TELEA)

    # 6. 去掉填充
    inpainted = inpainted[pad:-pad, pad:-pad]

    # 7. 细节平滑（可选）
    # smooth = cv2.edgePreservingFilter(inpainted, flags=1, sigma_s=60, sigma_r=0.4)

    # 8. 亮度微调（只在掩码区域）
    result = cv2.illuminationChange(inpainted, mask=mask,
                                    alpha=1.0, beta=2)

    cv2.imwrite("result_no_reflection_v2.jpg", result)
    return result

# ------------------- 主程序 -------------------
if __name__ == "__main__":
    # img_path = "1.jpg"                     # 待处理图片路径
    img_path = "orange.png"                     # 待处理图片路径
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"未找到图片 {img_path}")
    remove_reflection_v2(img)