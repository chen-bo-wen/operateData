import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
import json

# sets = ['train', 'test', 'val']
sets = ['train', 'val']
#H:异嗜性粒细胞，L:淋巴细胞，M:单核细胞
classes = ["H","L","M"]


def convert(size, box):

    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

        


def convert_annotation(image_id, id, label_set_dir, index_f="xml",single_class = False):
    in_file = open('../data/Annotations/%s.%s' % (image_id[:-4], index_f))
    out_file = open('%s/%s.txt' % (label_set_dir, image_id[:-4]), 'w',encoding='UTF-8')

    if index_f == "json":
        json_data = json.load(in_file)
        shapes = json_data["shapes"]
        w, h = json_data['imageWidth'], json_data['imageHeight']

        for obj in shapes:

            cls_id = obj["label"]
            id.append(cls_id)
            if cls_id == "19":
                print(in_file)
            xmlbox = obj['points']
            b = (float(xmlbox[0][0]),  # xmin
                 float(xmlbox[0][1]),  # ymin
                 float(xmlbox[1][0]),  # xmax
                 float(xmlbox[1][1]))  # ymax
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    elif index_f == "xml":
        # print(in_file)
        try:
            xml_data = ET.parse(in_file)
            root = xml_data.getroot()
            width = int(root.findtext("size/width"))
            height = int(root.findtext("size/height"))
            print(in_file)


            for obj in root.findall("object"):
                difficult = obj.findtext("difficult")
                class_name = obj.findtext("name")
                if class_name not in classes and int(difficult) == 1:
                    continue
                cls_id = classes.index(class_name)

                bbox = (float(obj.findtext("bndbox/xmin")),
                        float(obj.findtext("bndbox/ymin")),
                        float(obj.findtext("bndbox/xmax")),
                        float(obj.findtext("bndbox/ymax")))

                bb = convert((width, height), bbox)

                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')



        except UnicodeDecodeError as u:
            print(u)
            # f = open(in_file)
        # xml_text = in_file.read()
        # root = ET.fromstring(xml_text)
        # width = int(root.findtext("size/width"))
        # height = int(root.findtext("size/height"))



        # root = None
        # try:
        #     xml_data = ET.parse(in_file)
        #     root = xml_data.getroot()
        #     width = int(root.findtext("size/width"))
        #     height = int(root.findtext("size/height"))
        # except Exception as e:
        #     print(e,in_file)
            # xml_data = ET.parse(in_file)
        # xml_data = ET.parse(in_file)
        # root = xml_data.getroot()
        # width = int(root.findtext("size/width"))
        # height = int(root.findtext("size/height"))




# wd = getcwd()
# print(wd)
# Image_root = './data/images/'


# for image_set in sets:
#     if not os.path.exists('data/labels/'):
#         os.makedirs('data/labels/')
#     image_ids = open('data/ImageSets/%s.txt' % (image_set)).read().strip().split()
#     list_file = open('data/medical_ruler_%s.txt' % (image_set), 'w')
#     for image_id in image_ids:
#         list_file.write('data/images//%s\n' % (image_id))  # 数据路径，可在这里修改，存放在这里并写到txt文件中
#         convert_annotation(image_id, id)
#     list_file.close()
# print(id)
# print(sorted(set(id)))
# print(len(sorted(set(id))))

def xml_to_txt(single_class):
    id = []

    for image_set in sets:
        label_set_dir = '../data/labels/%s' % image_set
        if not os.path.exists(label_set_dir):
            os.makedirs(label_set_dir)
        image_ids = os.listdir(label_set_dir.replace("labels", "images"))
        for image_id in image_ids:
            convert_annotation(image_id, id, label_set_dir, index_f="xml",single_class=single_class)


def json_to_txt(single_class):
    id = []

    for image_set in sets:
        label_set_dir = 'data/labels/%s' % image_set
        if not os.path.exists(label_set_dir):
            os.makedirs(label_set_dir)
        image_ids = os.listdir(label_set_dir.replace("labels", "images"))
        list_file = open('data/math_express_%s.txt' % (image_set), 'w')
        for image_id in image_ids:
            convert_annotation(image_id, id, label_set_dir, index_f="json")


if __name__ == "__main__":
    single_class = False # 是否以单类别标签进行训练
    xml_to_txt(single_class)