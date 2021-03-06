from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES # [8,16,32]
        self.anchor_ratios = cfg.ANCHOR_RATIOS # [0.5,1,2]
        self.feat_stride = cfg.FEAT_STRIDE[0]

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True) # 输入维度din，输出维度512，3*3卷积核
        # 输入维度：self.din
        # 输出维度 512
        # 3*3的卷积核

        # define bg/fg classifcation score layer
        # 是前景/背景 分数层
        # 输出维度 计算
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)


        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)  # 1*1卷积核，改变维度用
        # 输入维度512，输出：（k=3*3=9）*2  ---- 【anchor个数*前景/背景】

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0) # 1*1卷积核，改变维度用
        # 输入维度512，输出：（k=3*3=9）*4 ----【anchor个数*(4是对每个anchor的regression微调)】

        # 讲了老半天，anchors是个啥，啥时候出现，怎么生成？还是不知道。【也就是说，cls和reg层在训练的时候，就自己开始训练了，根本不需要先生成anchor，因为anchor是定义好的已经存在的】
        # 前面我们知道，60×40的map上每个位置对应输入图像上的9个anchors，这9个anchors都是固定大小和尺寸的！
        # 我们知道这个就够了，因为整个rpn就是要训练来预测每个位置这些anchors的前背景分类得分和边框回归值。
        # 所以只需要在proposal_layer开始的时候根据经验生成这些anchors，最后在rpn末端对这些anchors进行调整、排序和筛除。【只是用来训练cls和reg层】

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)
        # 根据anchor回归值微调anchor的大小和位置，获得真正的proposals

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)
        # anchor_target_layer主要就是为了得到两个东西：
        # 1. 第一个东西是对应的一张图像生成的anchor的类别，在训练时需要赋予一定数量的正样本(前景)
        #       和一定数量的负样本(背景)，其余的需要全部置成 - 1，表示训练的时候会忽略掉
        # 2. 第二个东西是对于每一个anchor的边框修正，在进行边框修正loss的计算时，只有前景anchor会起作用，
        #       可以看到这是bbox_inside_weights和bbox_outside_weights在实现。
        #       非前景和背景anchor对应的bbox_inside_weights和bbox_outside_weights都为0。

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):

        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True) # 3*3*512 卷积核
        # get rpn classification score 【并行】
        # 获得前景/背景概率
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)

        # get rpn offsets to the anchor boxes【并行】
        # 获得anchor微调regression四个值
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        # 根据reg的四个值，对anchor进行微调
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        # 如果是训练模式：需要为每个anchor分配正/负样例标签、并选取等比例的正/负样例，进行训练，有一些IoU的限制
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))

            # compute classification loss 计算前/背景分类loss
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            rpn_label = rpn_data[0].view(batch_size, -1)


            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1)) #在这里先将label展开成one_hot向量，
            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep) # 在这里对应label中为-1值的位置排除掉score中的值，并且变成[-1,2]的形状方便计算交叉熵loss
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data) # 在这里留下label中的非-1的值，表示对应的anchor与gt的IoU在0.7以上

            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)    # Cross entropy error 在这里计算交叉熵loss
            fg_cnt = torch.sum(rpn_label.data.ne(0))

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

            # compute bbox regression loss 计算回归微调loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])

        return rois, self.rpn_loss_cls, self.rpn_loss_box
