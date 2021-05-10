from torch import nn
from models.resnet import ResNetN
from models.neck import FPN
from models.fcos_det_head import PulseHead
from models.postprocess import GetPulsar


class PulseDectector(nn.Module):

    def __init__(self, cfg):

        super(PulseDectector, self).__init__()
        self.backbone = ResNetN(cfg.backbone)
        if cfg.neck is not None:
            self.neck = FPN(cfg.neck)
        else:
            self.neck = None
        self.det_head = PulseHead(cfg.det_head)

        self.postprocess = GetPulsar(cfg.postprocess)
        self._init_weights()

    def _init_weights(self):
        self.backbone.init_weights()
        self.det_head.init_weights()
        self.neck.init_weights()

    def forward(self, img, targets=None, img_info=None, gt_points=None):
        '''

        :param img: tensor batchsize * 3 * H * W
        :param targets: list batchsize [ N * ]
        :param img_info: list batchsize [ list N [ dict(img_path, img_size)]]
        :return: loss tensor
        '''
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        # x list l[batchsize * C * H * W]
        if self.training:
            assert targets is not None
            losses = self.det_head(x, targets, img_info,gt_points=gt_points)
            return losses
        centerpoints, bbox_cls, bbox_reg = self.det_head(x)  # centerpoints: list l[W]  bbox_cls: list l[batchsize * 1 * 1 * W] bbox_reg: list l[batchsize * 2 * 1 * W]
        predict_pulsars = self.postprocess(centerpoints, bbox_cls, bbox_reg, img_info)
        return predict_pulsars
