import torch
import torch.nn.functional as F
from torch import nn

from .box_ops import BoxCoder, box_iou, process_box, nms
from .utils import Matcher, BalancedPositiveNegativeSampler


class RPNHead(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        # self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        # self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        # self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, 1)
        # 修改以便于阅读
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=1)

        for l in self.children():
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
    # 类别预测和边框回归并行
    def forward(self, x):
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        bbox_reg = self.bbox_pred(x)
        return logits, bbox_reg


class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head,
                fg_iou_thresh, bg_iou_thresh,
                num_samples, positive_fraction,
                reg_weights,
                pre_nms_top_n, post_nms_top_n, nms_thresh):
        super().__init__()

        self.anchor_generator = anchor_generator #锚框生成
        self.head = head #rpn head

        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True) #anchor与GT匹配
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction) #分配正负样本比例
        self.box_coder = BoxCoder(reg_weights) # 位置信息与需要学习的回归目标(锚框偏移)转换

        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1

    def create_proposal(self, anchor, objectness, pred_bbox_delta, image_shape):
        # nms处理前后保留的anchor数
        """ objectness:类别预测,为每个类别的概率，不是非0即1 """
        if self.training:
            pre_nms_top_n = self._pre_nms_top_n['training']
            post_nms_top_n = self._post_nms_top_n['training']
        else:
            pre_nms_top_n = self._pre_nms_top_n['testing']
            post_nms_top_n = self._post_nms_top_n['testing']

        pre_nms_top_n = min(objectness.shape[0], pre_nms_top_n) #获取保留数
        top_n_idx = objectness.topk(pre_nms_top_n)[1] #获取索引
        score = objectness[top_n_idx] #获取保留的anchor的置信度
        #获取proposal，输入锚框和偏移量，返回预测
        proposal = self.box_coder.decode(pred_bbox_delta[top_n_idx], anchor[top_n_idx])

        proposal, score = process_box(proposal, score, image_shape, self.min_size)
        keep = nms(proposal, score, self.nms_thresh)[:post_nms_top_n]
        proposal = proposal[keep]
        return proposal

    def compute_loss(self, objectness, pred_bbox_delta, gt_box, anchor):
        iou = box_iou(gt_box, anchor)
        label, matched_idx = self.proposal_matcher(iou)

        pos_idx, neg_idx = self.fg_bg_sampler(label)
        idx = torch.cat((pos_idx, neg_idx))
        # 仅使用正样本(回归损失仅使用正样本，分类损失使用正样本和负样本)
        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], anchor[pos_idx])
        # 分类损失(二元交叉熵损失)(loss = -[y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))])
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[idx], label[idx])
        # 边框回归损失(L1 loss)
        box_loss = F.l1_loss(pred_bbox_delta[pos_idx], regression_target, reduction='sum') / idx.numel()

        return objectness_loss, box_loss

    def forward(self, feature, image_shape, target=None):
        """ 首先生成锚框,之后使用rpn head进行预测,然后调换维度顺序以方便操作
        在创建proposal时需要进行位置信息的转换,再保留置信度高的的锚框
        最后计算损失需要计算iou,并得到匹配"""
        if target is not None:
            gt_box = target['boxes']
        anchor = self.anchor_generator(feature, image_shape)

        objectness, pred_bbox_delta = self.head(feature)
        # 调换顺序以方便操作,将NCHW(批大小,通道数,高度,宽度),转换成 NHWC(批大小、高度、宽度、通道数)
        objectness = objectness.permute(0, 2, 3, 1).flatten()
        pred_bbox_delta = pred_bbox_delta.permute(0, 2, 3, 1).reshape(-1, 4)

        proposal = self.create_proposal(anchor, objectness.detach(), pred_bbox_delta.detach(), image_shape)
        if self.training:
            objectness_loss, box_loss = self.compute_loss(objectness, pred_bbox_delta, gt_box, anchor)
            return proposal, dict(rpn_objectness_loss=objectness_loss, rpn_box_loss=box_loss)

        return proposal, {}