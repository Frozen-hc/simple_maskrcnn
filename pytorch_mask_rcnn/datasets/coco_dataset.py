import os
from PIL import Image

import torch
from .generalized_dataset import GeneralizedDataset
       
        
class COCODataset(GeneralizedDataset):
    def __init__(self, data_dir, split, train=False):
        super().__init__()
        from pycocotools.coco import COCO
        
        self.data_dir = data_dir #数据集路径
        self.split = split #数据集名称
        self.train = train # 是否训练
        
        # 注释文件路径
        ann_file = os.path.join(data_dir, "annotations/instances_{}.json".format(split))
        self.coco = COCO(ann_file) #生成coco对象
        self.ids = [str(k) for k in self.coco.imgs] # 获取图像键值
        
        # classes's values must start from 1, because 0 means background in the model
        self.classes = {k: v["name"] for k, v in self.coco.cats.items()} # 获取小类别名称
        
        checked_id_file = os.path.join(data_dir, "checked_{}.txt".format(split)) # 检查文件
        if train:
            if not os.path.exists(checked_id_file):
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()] # 高宽比
            self.check_dataset(checked_id_file)

    # 通过id得到图像(RGB)
    def get_image(self, img_id):
        img_id = int(img_id)
        img_info = self.coco.imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, "{}".format(self.split), img_info["file_name"]))
        return image.convert("RGB")
    
    @staticmethod
    # 将中心+高宽表示转换为左上右下
    def convert_to_xyxy(boxes): # box format: (xmin, ymin, w, h)
        x, y, w, h = boxes.T
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax)
        
    def get_target(self, img_id):
        img_id = int(img_id)
        ann_ids = self.coco.getAnnIds(img_id) #通过输入图片的id、类别的id、实例的面积、是否是人群来得到图片注释的id
        anns = self.coco.loadAnns(ann_ids) #通过id得到注释
        boxes = []
        labels = []
        masks = []

        #获得图片中所有锚框、标签、掩膜
        if len(anns) > 0:
            for ann in anns:
                boxes.append(ann['bbox'])
                labels.append(ann["category_id"])
                mask = self.coco.annToMask(ann) # 将注释中的分割转换为二值化掩膜
                mask = torch.tensor(mask, dtype=torch.uint8)
                masks.append(mask)

            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = self.convert_to_xyxy(boxes)
            labels = torch.tensor(labels)
            masks = torch.stack(masks)
            
        #创建字典
        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks)
        return target
    
    