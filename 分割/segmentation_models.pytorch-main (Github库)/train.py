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


class Dataset(BaseDataset):
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
    
    # # CamVid
    # CLASSES = [
    #     "sky",
    #     "building",
    #     "pole",
    #     "road",
    #     "pavement",
    #     "tree",
    #     "signsymbol",
    #     "fence",
    #     "car",
    #     "pedestrian",
    #     "bicyclist",
    #     "unlabelled",
    # ]

    def __init__(self, images_dir, masks_dir, classes=None, augmentation=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id).replace('jpg', 'png') for image_id in self.ids]

        # Always map background ('unlabelled') to 0
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
                # v: i + 1
                v: i
                for i, v in enumerate(self.class_values)
                if v != self.background_class
            }
        )
        
        print('class_map === ', self.class_map)

        self.augmentation = augmentation

    def __getitem__(self, i):
        # Read the image
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Read the mask in grayscale mode
        mask = cv2.imread(self.masks_fps[i], 0)

        # Create a blank mask to remap the class values
        mask_remap = np.zeros_like(mask)

        # Remap the mask according to the dynamically created class map
        for class_value, new_value in self.class_map.items():
            mask_remap[mask == class_value] = new_value

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask_remap)
            image, mask_remap = sample["image"], sample["mask"]
        
        image = image.transpose(2, 0, 1)
        return image, mask_remap

    def __len__(self):
        return len(self.ids)
    

# training set images augmentation
def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0
        ),
        # A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        # 设置大小可以被 32 整除
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0, always_apply=True), # border_mode=0 填充到黑色
        A.RandomCrop(height=320, width=320, always_apply=True),
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # A.PadIfNeeded(384, 480),
        A.PadIfNeeded(512, 512),
    ]
    return A.Compose(test_transform)


class CamVidModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, **kwargs):
        # print('encoder_weights ===', encoder_weights, type(encoder_weights))
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
        image, mask = batch

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
            tp, fp, fn, tn, reduction="micro-imagewise" # each image's IoU is calculated separately
        )
        # calculating the overall TP, FP, FN, TN across all categories at first, and then using these aggregate values to compute the global metric.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro") # 
        
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
        
    
def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument("--data-dir", type=str, default='/app/chen/segmentation_models.pytorch-main/data/VOC/')
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--arch", type=str, default="Unet", help="Model architecture.")
    parser.add_argument("--encoder", type=str, default="resnext101_32x8d", help="Encoder name.")
    parser.add_argument("--save-dirpath", type=str, default="saved-model-dir/unet/", help="save model path.")
    parser.add_argument("--save-filename-part", type=str, default="voc_best_model", help="save model filename (part).")
    parser.add_argument("--save-top-k", type=int, default=1, help="save model top_k.")
    parser.add_argument("--mode", type=str, default="max", help="save model mode.")
    parser.add_argument("--monitor", type=str, default="valid_dataset_iou", help="save model monitor.")
    parser.add_argument("--encoder-weights", type=str, default="imagenet", help="Encoder weights.")
    parser.add_argument("--classes", type=list, default="")
    args = parser.parse_args()

    x_train_dir = os.path.join(args.data_dir, "train")
    y_train_dir = os.path.join(args.data_dir, "trainannot")

    x_valid_dir = os.path.join(args.data_dir, "val")
    y_valid_dir = os.path.join(args.data_dir, "valannot")

    
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        augmentation=get_validation_augmentation(),
    )

    # Change to > 0 if not on Windows machine
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, num_workers=4)

    # Always include the background as a class
    OUT_CLASSES = len(train_dataset.CLASSES)

    model = CamVidModel(args.arch, args.encoder, args.encoder_weights, in_channels=3, out_classes=OUT_CLASSES)
    
    ###### 保存最优模型
    checkpoint_callback = ModelCheckpoint(
        monitor = args.monitor,  # 监控的指标
        dirpath = args.save_dirpath,  # 保存路径
        filename = '{epoch:03d}' + '_' + args.arch + '_' + args.encoder + '_' + str(args.epochs) + '_' + str(args.batch_size) + '_' + args.save_filename_part,  # 文件名
        save_top_k = args.save_top_k,
        mode = args.mode,  # 当监控的指标最大时保存
        # save_weights_only = True
    )

    # trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)
    trainer = pl.Trainer(max_epochs=args.epochs, 
                         log_every_n_steps=1, 
                         callbacks=[checkpoint_callback],
                         strategy='ddp_find_unused_parameters_true' # 使用：报错 RuntimeError: It looks like your LightningModule has parameters that were not used in producing the loss returned by training_step.
                         ) 

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

           

if __name__ == "__main__":
    main()
    

# CamVid

# FPN resnext50_32x4d


    # archs = [
    #     Unet,
    #     UnetPlusPlus,
    #     MAnet,
    #     Linknet,
    #     FPN,
    #     PSPNet,
    #     DeepLabV3,
    #     DeepLabV3Plus,
    #     PAN,
    # ]


# python train.py \
# --epochs=50 \
# --data-dir='/app/chen/segmentation_models.pytorch-main/data/VOC/' \
# --save-dirpath='saved-model-dir/Unet/' \
# --save-filename-part='VOC' \
# --batch-size=16 \
# --arch='Unet' \
# --encoder-weights='imagenet' \
# --encoder="dpn131"


#### VOC 
# python train.py \
# --epochs=100 \
# --data-dir='/app/chen/segmentation_models.pytorch-main/data/VOC/' \
# --save-dirpath='saved-model-dir/Unet_modify/' \
# --save-filename-part='VOC' \
# --batch-size=16 \
# --arch='Unet' \
# --encoder-weights='ssl' \
# --encoder="resnext101_32x16d"


#  strategy='ddp_find_unused_parameters_true'
###########