from collections import OrderedDict

import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc

from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer


class MaskRCNN(nn.Module):
    """
    Implements Mask R-CNN.

    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.
    输入图像为[C, H, W]大小的tensor,在0-1范围内

    The behavior of the model changes depending if it is in training or evaluation mode.
    模型在训练和评估模式下会不同

    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
    输入为图像、边框、标签、掩膜

    The model returns a Dict[Tensor], containing the classification and regression losses
    for both the RPN and the R-CNN, and the mask loss.
    返回类别预测、RPN和R-CNN的回归损失、掩膜损失

    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format,
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)
    预测时只需要输入图像，返回边框、标签、置信度、掩膜
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        主干网络：计算特征
        num_classes (int): number of output classes of the model (including the background).
        类别数
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        锚框和GT最小IoU阈值(前景)
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        锚框和GT最大IoU阈值(背景)
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn训练计算损失时采样的锚框数
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn训练的正样本比例
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        rpn编解码边框回归的权重
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        训练时在nms之前保留的proposal数
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        测试时在nms之前保留的proposal数
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        训练时在nms之后保留的proposal数
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        测试时在nms之后保留的proposal数
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        后处理rpn proposal的阈值

        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        训练分类head时proposal和GT的最小阈值(前景)
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        训练分类head时proposal和GT的最大阈值(背景)
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        分类head训练时采样的proposal数
        box_positive_fraction (float): proportion of positive proposals during training of the
            classification head
        分类head训练时正样本的比例
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        分类head编解码边框回归的权重
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        预测时返回proposal的最小置信度阈值
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        预测时预测头的nms阈值
        box_num_detections (int): maximum number of detections, for all classes.
        所有类别的最大检测数

    """

    def __init__(self, backbone, num_classes,
                # RPN parameters
                # FPN每层提取到的特征为2000个，多层加在一起经过nms后同样保留2000个
                rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                rpn_num_samples=256, rpn_positive_fraction=0.5,
                rpn_reg_weights=(1., 1., 1., 1.),
                rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                rpn_nms_thresh=0.7,
                # RoIHeads parameters
                box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                box_num_samples=512, box_positive_fraction=0.25,
                box_reg_weights=(10., 10., 5., 5.),
                box_score_thresh=0.1, box_nms_thresh=0.6, box_num_detections=100):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels

        #------------- RPN --------------------------
        anchor_sizes = (128, 256, 512) #锚框大小
        anchor_ratios = (0.5, 1, 2) # 锚框高宽比
        num_anchors = len(anchor_sizes) * len(anchor_ratios) # 锚框个数
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios) #生成锚框
        rpn_head = RPNHead(out_channels, num_anchors) #rpn head

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test) # 创建字典
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        #todo:为什么计算损失用的是匹配被认为是正样本的锚框和proposal
        self.rpn = RegionProposalNetwork(
                rpn_anchor_generator, rpn_head,
                rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                rpn_num_samples, rpn_positive_fraction,
                rpn_reg_weights,
                rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        #------------ RoIHeads --------------------------
        # shape[N, C, output_height, output_width]
        box_roi_pool = RoIAlign(output_size=(7, 7), sampling_ratio=2)

        resolution = box_roi_pool.output_size[0] #output_size=(7,7)
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        # 预测类别置信度以及边框偏移
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)

        self.head = RoIHeads(
                box_roi_pool, box_predictor,
                box_fg_iou_thresh, box_bg_iou_thresh,
                box_num_samples, box_positive_fraction,
                box_reg_weights,
                box_score_thresh, box_nms_thresh, box_num_detections)

        self.head.mask_roi_pool = RoIAlign(output_size=(14, 14), sampling_ratio=2)

        layers = (256, 256, 256, 256)
        dim_reduced = 256
        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers, dim_reduced, num_classes)

        #------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=800, max_size=1333,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225])

    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]

        image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]
        feature = self.backbone(image)

        proposal, rpn_losses = self.rpn(feature, image_shape, target)
        result, roi_losses = self.head(feature, proposal, image_shape, target)

        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result


#done
# 预测类别置信度以及边框偏移
class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """

        d = OrderedDict()
        next_feature = in_channels
        # enumerate第二参数代表开始的索引
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        d['mask_conv5'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['relu5'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')


class ResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        # resnet
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        #固定第一层卷积层和最后的全卷积层
        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # 以字典形式包装一组网络层，并去除了最后的全连接层
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256

        # self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        # self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        # 作了修改以便于阅读
        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, kernel_size=1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels,
                                            kernel_size=3, stride=1, padding=1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for module in self.body.values():
            x = module(x)
        # 最后加入1x1卷积和3x3卷积
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x


def maskrcnn_resnet50(pretrained, num_classes, pretrained_backbone=True, msd_dir=None):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """

    # todo:貌似无用
    if pretrained:
        backbone_pretrained = False

    backbone = ResBackbone('resnet50', pretrained_backbone) #使用resnet的backbone
    model = MaskRCNN(backbone, num_classes)

    if pretrained:
        if msd_dir:
            model_state_dict = torch.load(msd_dir)
        else:
            model_urls = {
                'maskrcnn_resnet50_fpn_coco':
                    'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
            }
            model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'], model_dir='../checkpoints')

        pretrained_msd = list(model_state_dict.values()) #msd: model state dict
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        # 删除backbone.fpn.inner_blocks.0~2与backbone.fpn.layer_blocks.0~2, 保留了backbone.fpn.inner_blocks.3
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        # 只加载一部分预训练的权重
        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        for i, name in enumerate(msd):
            if i in skip_list:
                continue
            msd[name].copy_(pretrained_msd[i])

        model.load_state_dict(msd)

    return model