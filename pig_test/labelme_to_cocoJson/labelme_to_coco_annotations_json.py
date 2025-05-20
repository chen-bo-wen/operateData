import json
import numpy as np
import glob
import PIL.Image
from PIL import ImageDraw
from shapely.geometry import Polygon

class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path='./new.json'):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.annotations = []
        self.images = []
        self.categories = [{"id": 1, "name": "1", "color": [244, 108, 59], "supercategory": ""}, 
                        {"id": 2, "name": "2", "color": [99, 102, 129], "supercategory": ""}, 
                        {"id": 3, "name": "3", "color": [249, 193, 0], "supercategory": ""}, 
                        {"id": 4, "name": "4", "color": [160, 180, 0], "supercategory": ""}, 
                        {"id": 5, "name": "5", "color": [115, 82, 59], "supercategory": ""}, 
                        {"id": 6, "name": "6", "color": [217, 213, 180], "supercategory": ""}, 
                        {"id": 7, "name": "7", "color": [51, 142, 137], "supercategory": ""}, 
                        {"id": 8, "name": "8", "color": [218, 147, 70], "supercategory": ""}, 
                        {"id": 9, "name": "9", "color": [234, 132, 163], "supercategory": ""}, 
                        {"id": 10, "name": "10", "color": [61, 127, 236], "supercategory": ""}, 
                        {"id": 11, "name": "11", "color": [81, 202, 147], "supercategory": ""}, 
                        {"id": 12, "name": "12", "color": [178, 171, 245], "supercategory": ""}, 
                        {"id": 13, "name": "13", "color": [42, 67, 196], "supercategory": ""}, 
                        {"id": 14, "name": "14", "color": [64, 158, 210], "supercategory": ""}, 
                        {"id": 15, "name": "15", "color": [206, 77, 194], "supercategory": ""}, 
                        {"id": 16, "name": "16", "color": [45, 86, 51], "supercategory": ""}, 
                        {"id": 17, "name": "17", "color": [139, 140, 69], "supercategory": ""}, 
                        {"id": 18, "name": "18", "color": [151, 218, 229], "supercategory": ""}, 
                        {"id": 19, "name": "19", "color": [135, 41, 60], "supercategory": ""}, 
                        {"id": 20, "name": "20", "color": [118, 33, 203], "supercategory": ""}, 
                        {"id": 21, "name": "21", "color": [42, 94, 246], "supercategory": ""}, 
                        {"id": 22, "name": "22", "color": [211, 109, 209], "supercategory": ""}, 
                        {"id": 23, "name": "23", "color": [66, 152, 210], "supercategory": ""}, 
                        {"id": 24, "name": "24", "color": [188, 189, 231], "supercategory": ""}, 
                        {"id": 25, "name": "25", "color": [68, 70, 179], "supercategory": ""}, 
                        {"id": 26, "name": "26", "color": [65, 168, 121], "supercategory": ""}, 
                        {"id": 27, "name": "27", "color": [245, 117, 90], "supercategory": ""}, 
                        {"id": 28, "name": "28", "color": [43, 126, 153], "supercategory": ""}, 
                        {"id": 29, "name": "29", "color": [248, 169, 168], "supercategory": ""}, 
                        {"id": 30, "name": "30", "color": [66, 145, 255], "supercategory": ""}, 
                        {"id": 31, "name": "31", "color": [249, 195, 28], "supercategory": ""}, 
                        {"id": 32, "name": "32", "color": [166, 219, 26], "supercategory": ""}, 
                        {"id": 33, "name": "33", "color": [147, 110, 255], "supercategory": ""}, 
                        {"id": 34, "name": "34", "color": [99, 102, 129], "supercategory": ""}, 
                        {"id": 35, "name": "35", "color": [225, 68, 134], "supercategory": ""}, 
                        {"id": 36, "name": "36", "color": [198, 129, 79], "supercategory": ""}, 
                        {"id": 37, "name": "37", "color": [84, 188, 203], "supercategory": ""}, 
                        {"id": 38, "name": "38", "color": [59, 112, 55], "supercategory": ""}, 
                        {"id": 39, "name": "39", "color": [123, 53, 137], "supercategory": ""}, 
                        {"id": 40, "name": "40", "color": [138, 188, 152], "supercategory": ""}] # 指定标注的类别
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.save_json()

    # 定义读取图像标注信息的方法
    def image(self, data, num):
        image = {}
        height = data['imageHeight']
        width = data['imageWidth']
        image['id'] = num + 1
        image['width'] = width
        image['height'] = height
        image['file_name'] = data['imagePath'].split('/')[-1]
        image['license'] = ""
        image['flickr_url'] = ""
        image['coco_url'] = ""
        image['date_captured'] = ""
        self.height = height
        self.width = width
        return image

    # 定义数据转换方法
    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, 'r') as fp:
                data = json.load(fp)  # 加载json文件
                self.images.append(self.image(data, num)) # 读取所有图像标注信息并加入images数组
                for shapes in data['shapes']:
                    label = shapes['label']
                    points = shapes['points']
                    shape_type = shapes['shape_type']
                    if shape_type == 'rectangle':
                        points = [points[0],[points[0][0],points[1][1]],points[1],[points[1][0],points[0][1]]]     
                    self.annotations.append(self.annotation(points, label, num)) # 读取所有检测框标注信息并加入annotations数组
                    self.annID += 1
        print(self.annotations)

    # 定义读取检测框标注信息的方法
    def annotation(self, points, label, num):
        annotation = {}
        annotation['id'] = self.annID
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
        annotation['category_id'] = self.getcatid(label)
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        poly = Polygon(points)
        area_ = round(poly.area, 6)
        annotation['area'] = area_
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        return annotation

    # 定义读取检测框的类别信息的方法
    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['categories'] = self.categories
        data_coco['images'] = self.images
        data_coco['annotations'] = self.annotations
        data_coco['info'] = ""
        data_coco['licenses'] = ""
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # 写入指定路径的json文件，indent=4 更加美观显示

labelme_json = glob.glob('pig_self_test/labelme_json/*.json')  # 获取指定目录下的json格式的文件
labelme2coco(labelme_json, 'pig_self_test/annotations_json/new.json') # 指定生成文件路径