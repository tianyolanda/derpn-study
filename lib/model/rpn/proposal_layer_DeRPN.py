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
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .generate_anchor_strings_DeRPN import generate_anchor_strings
# from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from .bbox_transform_DeRPN import bbox_transform_inv_DeRPN, clip_boxes, clip_boxes_batch

from model.nms.nms_wrapper import nms

from .anchorstring_to_proposal_DeRPN import anchorstring_to_proposal

import pdb

DEBUG = False

class _DeRPN_ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """
    # 生成anchors ：2w个大概
    # 根据reg和cls的值，筛选出300个proposal
    def __init__(self, feat_stride, w_an, h_an):
        super(_DeRPN_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchor_strings_w = torch.from_numpy(generate_anchor_strings(w_an = np.array(w_an))).float()
        self._anchor_strings_h = torch.from_numpy(generate_anchor_strings(w_an = np.array(h_an))).float()

        self._num_anchor_strings_w = self._anchor_strings_w.size(0)  # 7
        self._num_anchor_strings_h = self._anchor_strings_h.size(0)  # 7


        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        scores_w = input[0][:, self._num_anchors_w:, :, :] # [1,7,height,width]
        scores_h = input[1][:, self._num_anchors_h:, :, :] # [1,7,height,width]

        bbox_deltas_w = input[2]
        bbox_deltas_h = input[3]

        im_info = input[4]
        cfg_key = input[5] # Train / Test 标签

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N # Train: 12000, Test: 6000
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N # Train: 2000,  Test: 300
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH # Train: 0.7,  Test: 0.7
        min_size      = cfg[cfg_key].RPN_MIN_SIZE # Train: 8,  Test: 16
        combination_topN = cfg[cfg_key].DERPN_COM_TOP_N
        combination_topk = cfg[cfg_key].DERPN_COM_TOP_K

        batch_size = bbox_deltas_w.size(0)

        feat_height, feat_width = scores_w.size(2), scores_w.size(3) # feat_height 是featuer map的宽，feat_height是高

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

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        # reg 层的reshape
        bbox_deltas_w = bbox_deltas_w.permute(0, 2, 3, 1).contiguous()
        # permute()实现维度调转： 从 torch.Size([1, 7, 60, 40]) 变为 torch.Size([1, 60, 40, 7])
        bbox_deltas_w = bbox_deltas_w.view(batch_size, -1, 2)
        # 维度变换： 变为 torch.Size([1, 16800, 2])
        # '16800' 表示 anchors string 个数：60*40*7
        # '2' 表示 reg 由网络计算出的偏差值


        bbox_deltas_h = bbox_deltas_h.permute(0, 2, 3, 1).contiguous()
        bbox_deltas_h = bbox_deltas_h.view(batch_size, -1, 2)

        # Same story for the scores:
        # cls score层的reshape
        scores_w = scores_w.permute(0, 2, 3, 1).contiguous()  #  [batch_size,height,width,7]
        scores_w = scores_w.view(batch_size, -1)  # [batch_size, 16800]


        scores_h = scores_h.permute(0, 2, 3, 1).contiguous()
        scores_h = scores_h.view(batch_size, -1)  # [batch_size, 16800]

        # Convert anchors into proposals via bbox transformations
        # 将anchor进行regression变换,根据边框回归值调整边框的大小和位置获得真正的proposals，
        proposal_strings_w = bbox_transform_inv_DeRPN(anchor_strings_w, bbox_deltas_w, batch_size)
        proposal_strings_h = bbox_transform_inv_DeRPN(anchor_strings_h, bbox_deltas_h, batch_size)
        # 输出依然是([1, 16800, 2])

        # proposals combination 根据anchor string w,h 逐像素生成候选框：
        proposals_w = anchorstring_to_proposal(proposal_strings_w, proposal_strings_h,
                                               scores_w,scores_h,
                                               combination_topN, combination_topk, batch_size,self._num_anchor_strings_w)
        proposals_h = anchorstring_to_proposal(proposal_strings_h, proposal_strings_w,
                                               scores_h, scores_w,
                                               combination_topN, combination_topk, batch_size,self._num_anchor_strings_w)
        # 两者的并集，作为proposals
        proposals = torch.cat(proposals_w, proposals_h)

        scores = proposals[:,:,4] # 分数矩阵
        proposals = proposals[:,:,0:4] # proposal 矩阵

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)



        # 将超出输入图像边界的proposals裁剪至图像边界内，并将尺寸过小的proposals滤除掉
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it euqal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)
        
        # _, order = torch.sort(scores_keep, 1, True)


        # 对proposals进行得分排序，保留前2000个高分proposals，最后用nms过滤，保留最好的前300个proposals。
        scores_keep = scores
        proposals_keep = proposals
        _, order = torch.sort(scores_keep, 1, True)
        # _ ：排序后的score_keep
        # order ：排序后的下标标号

        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]
            scores_single = scores_keep[i]

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep
