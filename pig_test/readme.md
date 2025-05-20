1、使用 labelme 进行标注，得到标注文件，标注的标签在官方训练数据集上分成了 1~40 （其中27~~39。。）

2、使用 labelme_to_cocoJson 文件夹下的两个文件，将 labelme 的标注文件转为 coco 数据集格式的 json，两个 json 分别用于 CIEN 和 LWEN 的训练

3、可使用 annotations_json_on_png.py 和 Class_agnostic_annotations_json_on_png.py 将可标注文件可视化至原图

4、json_on_img_result 只是可视化标注文件至原图上的结果
