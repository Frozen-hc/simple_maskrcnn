import torch


class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between
            M ground-truth boxes and N predicted boxes.
            包含M个GT和N个锚框之间的IoU值

        Returns:
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
            返回标签以及匹配的GT编号
        """

        value, matched_idx = iou.max(dim=0) # 最大值iou以及索引
        label = torch.full((iou.shape[1],), -1, dtype=torch.float, device=iou.device) #创建一个[N,1]的全-1(忽略)tensor

        label[value >= self.high_threshold] = 1 # 高于正样本最低阈值的设为正样本
        label[value < self.low_threshold] = 0 # 低于负样本最高阈值的设为负样本

        if self.allow_low_quality_matches:
            highest_quality = iou.max(dim=1)[0] # 第一个GT与锚框匹配的最大iou
            gt_pred_pairs = torch.where(iou == highest_quality[:, None])[1]
            label[gt_pred_pairs] = 1

        return label, matched_idx


class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction

    def __call__(self, label):
        positive = torch.where(label == 1)[0]
        negative = torch.where(label == 0)[0]

        num_pos = int(self.num_samples * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.num_samples - num_pos
        num_neg = min(negative.numel(), num_neg)

        pos_perm = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        neg_perm = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[pos_perm]
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx


def roi_align(features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    if torch.__version__ >= "1.5.0":
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, False)
    else:
        return torch.ops.torchvision.roi_align(
            features, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio)


class AnchorGenerator:
    def __init__(self, sizes, ratios):
        self.sizes = sizes
        self.ratios = ratios

        self.cell_anchor = None
        self._cache = {}

    # 生成锚框模板
    def set_cell_anchor(self, dtype, device):
        if self.cell_anchor is not None:
            return
        sizes = torch.tensor(self.sizes, dtype=dtype, device=device)
        ratios = torch.tensor(self.ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(ratios) # r**0.5
        w_ratios = 1 / h_ratios # r**-0.5

        hs = (sizes[:, None] * h_ratios[None, :]).view(-1) #.view展平为一维向量
        ws = (sizes[:, None] * w_ratios[None, :]).view(-1)

        self.cell_anchor = torch.stack([-ws, -hs, ws, hs], dim=1) / 2

    #todo:计算预测特征图对应原始图像上的所有anchors的坐标
    def grid_anchor(self, grid_size, stride):
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        shift_x = torch.arange(0, grid_size[1], dtype=dtype, device=device) * stride[1] #对应原图上的x
        shift_y = torch.arange(0, grid_size[0], dtype=dtype, device=device) * stride[0]

        y, x = torch.meshgrid(shift_y, shift_x)
        x = x.reshape(-1)
        y = y.reshape(-1)
        shift = torch.stack((x, y, x, y), dim=1).reshape(-1, 1, 4)

        anchor = (shift + self.cell_anchor).reshape(-1, 4)
        return anchor

    #todo:将计算得到的所有anchors信息进行缓存
    def cached_grid_anchor(self, grid_size, stride):
        key = grid_size + stride
        #如果self._cache是字典类型直接返回
        if key in self._cache:
            return self._cache[key]
        anchor = self.grid_anchor(grid_size, stride)

        if len(self._cache) >= 3:
            self._cache.clear()
        self._cache[key] = anchor
        #返回所有预测特征层上的所有anchors坐标信息
        return anchor

    #todo:不懂
    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device
        grid_size = tuple(feature.shape[-2:]) #特征图大小
        stride = tuple(int(i / g) for i, g in zip(image_size, grid_size)) #计算特征层上的一步(一个cell)对应原始图像上的步长

        self.set_cell_anchor(dtype, device)

        anchor = self.cached_grid_anchor(grid_size, stride)
        return anchor