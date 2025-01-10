import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import os
import cv2
import numpy as np
import albumentations as A
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import shutil
from tqdm import tqdm  # 导入 tqdm 库
from matplotlib.colors import ListedColormap
from PIL import Image

# import sys
# print('sys.executable', sys.executable) # /home/huazhi/anaconda3/envs/seg_pytroch/bin/python


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


### 加载数据
class Dataset(BaseDataset):
    # CamVid数据集中用于图像分割的所有标签类别
    # CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
    #            'tree', 'signsymbol', 'fence', 'car',
    #            'pedestrian', 'bicyclist', 'unlabelled']
    
    # VOC
    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
        # "unlabelled",
    ]

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None):
        self.ids = os.listdir(images_dir) # 所有图片文件名
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id).replace('jpg', 'png') for image_id in self.ids]

        # Always map background ('unlabelled') to 0
        # self.background_class = self.CLASSES.index("unlabelled")
        self.background_class = self.CLASSES.index("background")

        # If specific classes are provided, map them dynamically
        if classes:
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        else:
            self.class_values = list(range(len(self.CLASSES)))  # Default to all classes

        # Create a remapping dictionary: class value in dataset -> new index (0, 1, 2, ...)
        # Background will always be 0, other classes will be remapped starting from 1.
        self.class_map = {self.background_class: 0}
        self.class_map.update(
            {
                v: i
                # v: i + 1
                for i, v in enumerate(self.class_values)
                if v != self.background_class
            }
        )
        
        print('self.class_map ===', self.class_map)

        self.augmentation = augmentation


    def __getitem__(self, i):
        # Read the image
        image = cv2.imread(self.images_fps[i]) # 使用 OpenCV 读取图像，默认为 BGR 格式
        image_name = self.ids[i] # get file name
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Read the mask in grayscale mode
        mask = cv2.imread(self.masks_fps[i], 0)

        # Create a blank mask to remap the class values
        mask_remap = np.zeros_like(mask)
        
        #### 读取 mask 的时候能否统一颜色。。。
        # Remap the mask according to the dynamically created class map
        for class_value, new_value in self.class_map.items():
            mask_remap[mask == class_value] = new_value

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]
        image = image.transpose(2, 0, 1)
        return image, mask_remap, image_name


    def __len__(self):
        return len(self.ids)


class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            encoder_weights=encoder_weights,
            classes=out_classes,
            **kwargs,
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained=encoder_weights)
        self.number_of_classes = out_classes
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for multi-class segmentation
        self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)

        # Step metrics tracking
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        # Normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        # print('batch===', batch)
        image, mask, filename = batch # 在 Dataset类的__getitem__ 中有 return image_name

        # Ensure that image dimensions are correct
        assert image.ndim == 4  # [batch_size, channels, H, W]

        # Ensure the mask is a long (index) tensor
        mask = mask.long()

        # Mask shape
        assert mask.ndim == 3  # [batch_size, H, W]

        # Predict mask logits
        logits_mask = self.forward(image)

        assert (
            logits_mask.shape[1] == self.number_of_classes
        )  # [batch_size, number_of_classes, H, W]

        # Ensure the logits mask is contiguous
        logits_mask = logits_mask.contiguous()

        # Compute loss using multi-class Dice loss (pass original mask, not one-hot encoded)
        loss = self.loss_fn(logits_mask, mask)

        # Apply softmax to get probabilities for multi-class segmentation
        prob_mask = logits_mask.softmax(dim=1)

        # Convert probabilities to predicted class labels
        pred_mask = prob_mask.argmax(dim=1)

        # Compute true positives, false positives, false negatives, and true negatives
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode="multiclass", num_classes=self.number_of_classes
        )

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage): # 输入的是 batch, "train"
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Per-image IoU and dataset IoU calculations
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        # 输出 iou 的值
        metrics = {
            f"{stage}_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True) # 记录的指标会保存在 lightning_logs 中

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
   
# ---------------------------------------------------------------
### 图像增强
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # A.PadIfNeeded(384, 480),
        A.PadIfNeeded(512, 512),
    ]
    return A.Compose(test_transform)



def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--data-dir", type=str, default='/app/chen/segmentation_models.pytorch-main/data/VOC/')
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--arch", type=str, default="Unet", help="Model architecture.")
    parser.add_argument("--encoder", type=str, default="resnext101_32x8d", help="Encoder name.")
    parser.add_argument("--encoder-weights", type=str, default="imagenet", help="Encoder weights.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-imgpath", type=str, default="visualize_images/", help="save model path.")
    parser.add_argument("--num-classes", type=int, required=True)
    # parser.add_argument("--classes", type=list, default=None)
    args = parser.parse_args()

    # 测试集
    x_valid_dir = os.path.join(args.data_dir, "val")
    y_valid_dir = os.path.join(args.data_dir, "valannot")

    x_test_dir = os.path.join(args.data_dir, "test")
    y_test_dir = os.path.join(args.data_dir, "testannot")
    
    
    # model = CamVidModel(args.arch, args.encoder, in_channels=3, out_classes=args.num_classes) # 加载模型实例
    # # state_dict = torch.load('/app/chen/segmentation_models.pytorch-main/saved-model-dir/unet/CamVid_best_model-v1.ckpt')
    # checkpoint = torch.load(args.checkpoint) # 加载相应权重
    # state_dict = checkpoint['state_dict']  # 只提取模型的权重
    # model.load_state_dict(state_dict)
    
    
    # CLASSES = ['car']
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
    )
    
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        # classes = args.classes
        # classes = CLASSES
    )
    
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    
    #####################
    model = CamVidModel.load_from_checkpoint(args.checkpoint,
                                            arch=args.arch,
                                            encoder_name=args.encoder,
                                            encoder_weights=args.encoder_weights,
                                            in_channels=3,
                                            out_classes=args.num_classes)
    trainer = pl.Trainer(devices=1) # 使用单个设备进行评估
    valid_metrics = trainer.validate(model, dataloaders=valid_loader, verbose=False)
    print('valid_metrics', valid_metrics)
    
    test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False) # pl lighting, Automatically set the model to evaluation mode
    print('test_metrics', test_metrics)

    # ---------------------------------------------------------------
    

    save_path = args.save_imgpath

    # 获取文件夹路径
    folder_path = os.path.dirname(save_path)
    
    
    # 删除文件夹里的已有文件，以保存重新预测的结果
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    else:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    # 删除文件
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    
    # 创建一个迭代器
    data_iterator = iter(test_loader)
    
    
    # 自动生成颜色映射
    # num_classes = args.classes  # 计算类别数量
    # colors = plt.cm.get_cmap('tab10', num_classes)  # 使用 'tab10' colormap 或其他可用的 colormap
    # color_map = {i: colors(i)[:3] for i in range(num_classes)}  # 仅提取 RGB 值
    
    # print('color map', color_map)
    color_mapping = {
            0: [0, 0, 0],        # black
            1: [255, 0, 0],      # red 
            2: [0, 0, 255],      # Blue1 
            3: [225, 255, 0],    # Yellow1    
            4: [0, 255, 0],      # Green1
            5: [160, 32, 240],   # purple  
            6: [255, 52, 179],   # Maroon1 
            7: [255, 165, 0],    # Orange  
            8: [30, 144, 255],   # DodgerBlue1 
            9: [169, 169, 255],   
            10: [255, 0, 165],     
            11: [255, 165, 165],   
            12: [128, 128, 0],     
            13: [0, 128, 128],     
            14: [128, 0, 128],     
            15: [255, 255, 128],   
            16: [128, 255, 255],   
            17: [255, 128, 255],   
            18: [128, 128, 128],   
            19: [64, 0, 0],        
            20: [255, 255, 255]  
        }

    # 遍历所有批次
    for _ in range(len(test_loader)):
        images, masks, image_names = next(data_iterator) # 所有的都是存储在一个批次里，放置在一个括号里
        # 处理每个批次的数据
        # print(images.shape, masks.shape, image_names)

        # Switch the model to evaluation mode
        with torch.no_grad():
            model.eval()
            logits = model(images)  # Get raw logits from the model

        # Apply softmax to get class probabilities
        # Shape: [batch_size, num_classes, H, W]
        
        pr_masks = logits.softmax(dim=1)
        # Convert class probabilities to predicted class labels
        pr_masks = pr_masks.argmax(dim=1)  # Shape: [batch_size, H, W] 
        
        ############# 如何生成 overlay 的图片
        # Visualize a few samples (image, ground truth mask, and predicted mask)
        for idx, (image, gt_mask, pr_mask, image_name) in tqdm(enumerate(zip(images, masks, pr_masks, image_names)), total=len(images), desc="Processing images"):

            if idx <= len(images): 
                gt_colored_mask = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
                gt_unique_classes = np.unique(gt_mask) # 格式如 [ 0  1  4  6  7  8  9 11 12 15 16 18 19]
                
                pr_colored_mask = np.zeros((*pr_mask.shape, 3), dtype=np.uint8) 
                pr_unique_classes = np.unique(pr_mask)
                
                for category in gt_unique_classes: # 
                    if category == 0:  # background
                        continue
                    gt_colored_mask[gt_mask == category] = color_mapping[category]
                
                for category in pr_unique_classes:
                    if category == 0:  
                        continue
                    # color = np.random.randint(0, 255, size=3)  # 随机颜色
                    pr_colored_mask[pr_mask == category] = color_mapping[category]

                # 5. 创建彩色 mask 图像
                gt_colored_mask_image = Image.fromarray(gt_colored_mask)
                pr_colored_mask_image = Image.fromarray(pr_colored_mask)
                
                # image is tensor，image.shape is (3, 512, 512), numpy can not process the data on gpu
                image_array = image.cpu().numpy().transpose(1, 2, 0)  # 转换为 (H, W, C) 格式
                image_array2 = (image_array * 255).astype(np.uint8) # 确保数据类型为 uint8
                image = Image.fromarray(image_array2)
                
                # # 6. 设置透明度
                gt_colored_mask_image.putalpha(160)
                pr_colored_mask_image.putalpha(160) 

                # combine 2 images，can not be tensor
                gt_combined_image = Image.alpha_composite(image.convert('RGBA'), gt_colored_mask_image.convert('RGBA'))
                pr_combined_image = Image.alpha_composite(image.convert('RGBA'), pr_colored_mask_image.convert('RGBA'))

                # 8. 保存叠加后的图像
                # image_name.replace('jpg','png') # png 格式支持保存透明度
                # combined_filename = os.path.join(folder_path, f'overlay_{image_name}')
                # pr_combined_image.save(combined_filename, 'PNG')
                # pr_colored_mask_image.save(combined_filename)
                
                ####### 当将真实的预测图从灰度图转为RGB图时，颜色如何设置对应的物体就显示为对应的图片
                plt.figure(figsize=(15, 6)) 

                # Original Image
                plt.subplot(1, 3, 1)
                # plt.imshow(
                #     image.cpu().numpy().transpose(1, 2, 0)
                # )  # Convert CHW to HWC for plotting
                plt.imshow(
                    image_array
                )  # Convert CHW to HWC for plotting
                plt.title("Image")
                plt.axis("off")
                

                # Ground Truth Mask
                plt.subplot(1, 3, 2)
                # plt.imshow(gt_mask.cpu().numpy(), cmap="tab20")  # Visualize ground truth mask
                plt.imshow(gt_combined_image)  # Visualize ground truth mask
                plt.title("Ground truth")
                plt.axis("off")

                # Predicted Mask
                plt.subplot(1, 3, 3)
                # plt.imshow(pr_mask.cpu().numpy(), cmap="tab20")  # Visualize predicted mask
                plt.imshow(pr_combined_image)  # Visualize predicted mask
                plt.title("Prediction")
                plt.axis("off")
                
                
                # # overlay
                # plt.subplot(1, 3, 3)
                # plt.imshow(overlay)  # Visualize predicted mask
                # plt.title("overlay")
                # plt.axis("off")

                # Save the figure
                # if not os.path.exists(os.path.join(folder_path, 'combine')):
                #     os.makedirs(os.path.join(folder_path, 'combine'))
                path = os.path.join(folder_path, f'{image_name}')
                plt.savefig(path, bbox_inches='tight', pad_inches=0)
                plt.close()  # Close the figure to free up memory
                
                ## save original image
                # if not os.path.exists(os.path.join(folder_path, 'image')):
                #     os.makedirs(os.path.join(folder_path, 'image'))
                # original_image_path = os.path.join(folder_path, 'image/', f'{image_name}')
                # # plt.imsave(original_image_path, image.cpu().numpy().transpose(1, 2, 0))
                # plt.imsave(original_image_path, image_array)

                # # save mask
                # image_name.replace('jpg', 'png')
                # if not os.path.exists(os.path.join(folder_path, 'mask')):
                #     os.makedirs(os.path.join(folder_path, 'mask'))
                # predicted_mask_path = os.path.join(folder_path, 'mask/', f'mask_{image_name}')
                # plt.imsave(predicted_mask_path, pr_mask.cpu().numpy(), cmap='tab20')
            else:
                break
        
    print('完成测试')
    
    
# ---------------------------------------------------------------
if __name__ == '__main__':
    main()
    
    # 最后计算 平均 test



# 适用于 modify 之后的模型，将CLASS标签的顺序修正之后

#### VOC
# python visualize2.py \
# --checkpoint='/app/chen/segmentation_models.pytorch-main/saved-model-dir/Unet/Unet_efficientnet-b7_100_16_VOC.ckpt' \
# --num-classes=21 \
# --save-imgpath='visualization/test_ccc/' \
# --data-dir='/app/chen/segmentation_models.pytorch-main/data/VOC/' \
# --batch-size=16 \
# --arch='Unet' \
# --encoder="efficientnet-b7"


# python visualize2.py \
# --checkpoint='/app/chen/segmentation_models.pytorch-main/saved-model-dir/Unet_modify/epoch=093_Unet_densenet161_100_16_VOC.ckpt' \
# --num-classes=21 \
# --save-imgpath='visualization/epoch=093_Unet_densenet161_100_16_VOC/' \
# --data-dir='/app/chen/segmentation_models.pytorch-main/data/VOC/' \
# --batch-size=16 \
# --arch='Unet' \
# --encoder-weights='imagenet' \
# --encoder="densenet161"


# epoch=093_Unet_densenet161_100_16_VOC         vgg16_bn
# epoch=093_Unet_densenet161_100_16_VOC       densenet161

