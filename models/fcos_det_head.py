import torch
import math
import torch.nn.functional as F
from torch import nn
from .layers.fcos_loss import FCOSLossComputation


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class PulseHead(nn.Module):

    def __init__(self, cfg):

        super(PulseHead,self).__init__()
        self.num_classes = 2
        self.fpn_strides = [4,8,16,32]
        self.norm_reg_targets = None
        self.loss_func = FCOSLossComputation(cfg)

        cls_tower = []
        bbox_tower = []
        for i in range(cfg['num_convs']):
            cls_tower.append(
                nn.Conv2d(
                    cfg['in_channels'],
                    cfg['cls_channels'],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32,cfg['cls_channels']))
            cls_tower.append(nn.ReLU(inplace=True))

            bbox_tower.append(
                nn.Conv2d(
                    cfg['in_channels'],
                    cfg['reg_channels'],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32,cfg['reg_channels']))
            bbox_tower.append(nn.ReLU(inplace=True))
        # cls_tower.append(nn.AdaptiveMaxPool2d((1, None)))
        # bbox_tower.append(nn.AdaptiveMaxPool2d((1,None)))
        # cls_tower.append(nn.MaxPool2d(()))
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            cfg['cls_channels'], self.num_classes, kernel_size=(3,3),
            stride=1, padding=1
        )
        self.bbox_pred= nn.Conv2d(
            cfg['reg_channels'], 4, kernel_size=(3,3),
            stride=1, padding=1
        )

    def init_weights(self):
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x, targets=None, img_info=None):
        logits = []
        bbox_reg = []

        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            bbox_pred = self.scales[l](self.bbox_pred(box_tower))

            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        locations=self.compute_locations(x) # l[w]
        if self.training:
            loss_box_cls, loss_box_reg = self.loss_func(locations, logits, bbox_reg, targets)
            losses = {
                'loss_cls': loss_box_cls,
                'loss_reg': loss_box_reg
            }
            return losses
        return locations, logits, bbox_reg # logits: l[N * C * 1 *W] bbox_reg: l[N * 2 * 1 * W]

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h,  w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        # locations = shift_x + stride//2
        return locations

