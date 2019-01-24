from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchor_strings_DeRPN import generate_anchor_strings

from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer_DeRPN(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
        给ground truth分配anchor，生成anchor的 分类标签 和 回归box
    """
    def __init__(self, feat_stride,  w_an, h_an):
        super(_AnchorTargetLayer_DeRPN, self).__init__()

        self._feat_stride = feat_stride
        # self._scales = scales
        # anchor_scales = scales
        # self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        # 生成9个anchor：shape: [9,4]
        # 生成了一张图，存储了9个anchors的对角坐标
        # self._num_anchors = self._anchors.size(0)
        self._feat_stride = feat_stride
        self._anchor_strings_w = torch.from_numpy(generate_anchor_strings(w_an = np.array(w_an))).float()
        self._anchor_strings_h = torch.from_numpy(generate_anchor_strings(w_an = np.array(h_an))).float()

        self._num_anchor_strings_w = self._anchor_strings_w.size(0)  # 7
        self._num_anchor_strings_h = self._anchor_strings_h.size(0)  # 7

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # 对于每个位于i的宽高为w，h的点，围绕点i生成9个anchor box。
        # 对9个anchor应用 predicted bbox deltas
        # 筛选出超出图像边框的anchor


        # input是这四个东西的组合！！！：(rpn_cls_score.data, gt_boxes, im_info, num_boxes)，下面将它们逐个拆解
        scores_w = input[0][:, self._num_anchors_w:, :, :] # [1,7,height,width]
        scores_h = input[1][:, self._num_anchors_h:, :, :] # [1,7,height,width]

        gt_boxes = input[1]
        im_info = input[2]
        num_boxes = input[3]

        # map of shape (..., H, W)
        height, width = scores_w.size(2), scores_w.size(3)
        # 这个是feature map长和宽

        batch_size = gt_boxes.size(0)

        feat_height, feat_width = scores_w.size(2), scores_w.size(3)

        # 下面是在原图上生成anchor
        # shift_x = np.arange(0, feat_width) * self._feat_stride #shape: [width,]
        # shift_y = np.arange(0, feat_height) * self._feat_stride #shape: [height,]
        # shift_x, shift_y = np.meshgrid(shift_x, shift_y)  #生成网格 shift_x shape: [height, width], shift_y shape: [height, width]
        # shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
        #                           shift_x.ravel(), shift_y.ravel())).transpose())
        # shifts = shifts.contiguous().type_as(rpn_cls_score).float() # shape[height*width, 4]
        #
        # # add A anchors (1, A, 4) to
        # # cell K shifts (K, 1, 4) to get
        # # shift anchors (K, A, 4)
        # # reshape to (K*A, 4) shifted anchors
        # A = self._num_anchors # A = 9
        # K = shifts.size(0) # K=height*width(特征图上的)
        #
        # self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
        # all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)  #shape[K,A,4] 得到所有的anchor
        # all_anchors = all_anchors.view(K * A, 4)  # 生成全部anchor (9*h*w,4)

        shift_x = np.arange(0, feat_width) * self._feat_stride  # [0,1,..,feat_width_w] * 16
        shift_y = np.arange(0, feat_height) * self._feat_stride  # shape (40,)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # 变成格子状（x是横着摞起来，y是竖着摞起来）
        # 此时 shift_x shape (60, 40)， shift_y shape (60, 40)

        shifts_w = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  )).transpose())
        # shift_x.ravel() 是把shift_x 抻平，类似flatten(),变成了shape (2400,)
        # vstack 是把2个(2400，)， 垂直（按照行顺序）的把数组给堆叠起来。shape：torch.Size([2, 2400])
        # transpose 是转置： 输出shape：torch.Size([2400, 2])

        shifts_w = shifts_w.contiguous().type_as(scores_w).float()# 调用view之前最好先contiguous #  x.contiguous().view()

        # tensor_1.type_as(tensor_2), 将tensor_1转换成tensor_2
        # 类型是指int float这些
        # 其余shape不变，依然是[2400,4]

        A = self._num_anchor_strings_w # 7
        K = shifts_w.size(0) # 2400

        self._anchor_strings_w = self._anchor_strings_w.type_as(scores_w) # 更改type和scores一样
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchor_strings_w = self._anchor_strings_w.view(1, A, 2) + shifts_w.view(K, 1, 2)
        anchor_strings_w = anchor_strings_w.view(1, K * A, 4).expand(batch_size, K * A, 2)
        # torch.Size([1, 16800, 2])
        # 对feature map 2400个点，每个点都生成7个anchor strings--(x,w)，共16800个 anchor strings
        # '2' 表示 reg 每个格子的生成值: (xmin,xmax)

        anchor_strings_h = anchor_strings_w # 将w的anchor赋值给h，两者是一样的

        # 这里为什么不需要batch_size,是因为每个batch，每张图片，生成的anchors都是一样的！！！
        # 所以生成一次就可以了！以及下面的计算合法anchor数量，这些操作都可以只做一次
        # 生成anchor后，对待每一张图片的gt_box，都计算所有anchor和t_box的overlap，然后计算两者的regression差值，存入bbox_targets

        total_anchors = int(K * A) # total_anchors记录anchor的数目

        # all_anchor 是一个[K * A,4]的tensor，K * A表示的是所有anchor的总数。4 表示的是 每个anchor里面的四个坐标
        # 下面keep的操作：对于每个anchor，判断它四个坐标，必须每一个都符合要求（在界限内），一旦有一个不符合，就把这个anchor置为零
        # keep 最后生成的是一个大小为[K * A]一维的向量，由0/1组成，表示对【所有anchor】是否符合要求的判断。0表示该向量不符合要求，1表示符合要求
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &  #  取第一列所有元素 xmin >=0
                (all_anchors[:, 1] >= -self._allowed_border) &  #  取第二列所有元素 ymin >=0
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &  #  取第三列所有元素 xmax <= width
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))  #  取第四列所有元素 ymax <= height

        # inds_inside 是将符合要求的anchor【编号】都提取出来，组成一个一维向量[ ]，大小不一定！！！
        inds_inside = torch.nonzero(keep).view(-1)
        #  inds_inside.size(0) 就是【合法的anchor的数量】

        # anchors保存了all_anchors中符合要求的所有anchor的四个坐标值。 keep only inside anchors
        # anchors的维度all_anchors相似，只不过剔除掉了不合法的anchor：
        # anchors：[batch_size(也许有), 有效anchor个数, 4]
        anchors = all_anchors[inds_inside, :]  # 在这里选出合理的anchors，指的是没超出边界的

        # label: 1 is positive, 0 is negative, -1 is dont care
        # 新建三个tensor，维度均为[batch_size, 合法anchor的数目N]: [batch_size,N]
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1) #先用-1填充labels
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()

        # 对所有的合法anchor, 计算其和gt_boxes的overlap，得到的shape: [len(anchors), len(gt_boxes)]
        # anchors: (N, 4) ndarray of float
        # gt_boxes: (b, K, 5) ndarray of float, K : h*w？？还是有几个gt，K就是几？问号

        overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        # overlaps是一个[batch_size, N, K]的矩阵，其中每个值表示N个anchor和K个gt的IoU。
        # 其中如果gt的面积为0，则值为0，如果anchor的面积为0，则值为-1

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2) # 对于每个anchor，找到最大的对应的gt_box的IoU大小和gt索引值。shape: [len(anchors),]
        # torch.max(a,2) 返回每个batch中第2维中最大值的那个元素值，且返回其索引。也就是：找到每个anchor和哪个gt的IoU最大，返回IoU值和索引值
        # max_overlaps：torch.Size([1, N])： N 表示N个anchor，每个和k个gt中的最大IoU【值】
        # argmax_overlaps：torch.Size([1, N])： N 表示N个anchor，与其有着最大IoU的gt的【编号】

        gt_max_overlaps, _ = torch.max(overlaps, 1) #对于每个gt，找到最大的overlap的anchor 的IoU大小和gt索引值: [len(anchors)]（和上面的正好是反着）
        # torch.max(a,1) 返回每个batch中第N维中最大值的那个元素值，且返回其索引。也就是：找到每个gt和哪个anchor的IoU最大，返回IoU值和索引值
        # gt_max_overlaps：torch.Size([1, k])： k 表示k个gt，每个和N个anchor中的最大IoU【值】
        # _: 编号

        # negative
        # If an anchor statisfied by positive and negative conditions set to negative
        # __C.TRAIN.RPN_CLOBBER_POSITIVES = False
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < 0.3] = 0  # If the max overlap of a anchor from a ground truth box is lower than this thershold, it is marked as background.
            # max_overlaps小于0.3的，其对应labels里的位置置为0：负例（目前，其余是-1）

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        # gt_max_overlaps等于0的，用1e-5表示。

        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)
        # 将gt_max_overlaps扩张维度(列数重复N次)：a = gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)
        # 找overlaps和gt_max_overlaps相同的值：b = overlaps.eq(a)【这步目的是，找到每一个gt对应的最大IoU的anchor】
        # 计算每一个anchor和几个gt是该gt的最大IoU。比如第m个anchor是gt1的最大IoU anchor，同时也是gt2的最大IoU anchor，那[batch_size,m]=2
        # keep输出是一个[batch_size,N]的tensor

        if torch.sum(keep) > 0:
            labels[keep > 0] = 1
            # 将这些anchor的label设置为1：正例

        # fg label: above threshold IOU
        # positive
        labels[max_overlaps >= 0.7] = 1  # Threshold used to select if an anchor box is a good foreground box (Default: 0.7)
        # anchor和某个gt的IoU大于0.7：正例

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < 0.3] = 0
            # anchor和某个gt的IoU小于0.3：负例

        # Total number of examples
        # __C.TRAIN.RPN_BATCHSIZE = 256
        num_fg = int(0.5 * 256)  # fraction of the batch size that is foreground anchors

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)
        # 计算正负例的数量

        for i in range(batch_size):
            # 这个操作需要针对每个batch来做，因为每张图片的正负例的数量不一样，有的多于需求，有的少于需求。

            # 正例筛选
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)
                # 返回label=1的元素的索引：一维向量，长度不定，因为不一定有几个正例
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                # permutation：打乱fg_inds顺序。不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]  # 随机选取要扔掉的正例
                labels[i][disable_inds] = -1  # 多余的正例，设置为不关心：label = -1

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = 256 - torch.sum((labels == 1).int(), 1)[i]  # Total number of background and foreground anchors

            # 负例筛选
            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1
        # for循环结束


        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        # 创造tensor([0, 4, 8])【当batch_size=3，gt_boxes.size(1)=4】

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        # tensor([[1, 0, 2, 1, 2, 0],
        #         [0, 4, 6, 0, 6, 4],
        #         [8, 0, 0, 8, 0, 8]])
        # argmax_overlaps维度：(batch_size, K)
        # 行表示第几张图片(第几个batch)，列表示第几个anchor，与其有着最大IoU的gt的【编号】

        # 这一步在算regression值！！bbox_targets是在计算每个anchor和对应的gt_box的regression差值
        # anchors: (N, 4) ndarray of float
        # gt_boxes: (b, K, 5) ndarray of float, K : h*w？？还是有几个gt，K就是几？问号
        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))
        # gt_boxes.view(-1,5): 把每个batch展开，维度变成 [batch_size*K, 5]
        # argmax_overlaps.view(-1)：按每个batch展开（之前已经做过，每个batch都会加上K个偏移），维度变成 [batch_size*K]
        # [argmax_overlaps.view(-1), :] 除掉了0元素？只留下有argmax_overlap



        # use a single value instead of 4 values for easy index.
        # Deprecated (outside weights)
        # __C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
        # # Give the positive RPN examples weight of p * 1 / {num positives}
        # # and give negatives a weight of (1 - p)
        # # Set to -1.0 to use uniform example weighting

        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]  # 1.0

        # Set to -1.0 to use uniform example weighting
        # __C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0
        # 正负例的权值一样
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            num_examples = torch.sum(labels[i] >= 0)
            positive_weights = 1.0 / num_examples.item()
            negative_weights = 1.0 / num_examples.item()
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        # bbox_outside_weights外部权重，目前负例的外部权重 = 正例的外部权重 = np.ones((1, 4)) * 1.0 / np.sum(labels >= 0)
        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights

        # 重新排布下面四个值
        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)  #

        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    # 将data的子集映射回原始data的size （count）
    # 将 data，变换为
    # 维度为 [batch_size, count]， fill为0，type为data 的新tensor
    # 其中count层的 inds = data
    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
     # 计算图像target的边界框回归
    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
