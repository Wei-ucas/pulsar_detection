import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class FPN(nn.Module):

    def __init__(self, cfg):
        super(FPN, self).__init__()
        out_channel = cfg['out_channels']
        in_channels = cfg['in_channels']
        self.out_channels = out_channel
        self.num_out = cfg['num_out']
        self.num_in = cfg['num_in']
        self.out_levels = cfg['out_levels']

        assert len(in_channels) == self.num_in and self.num_in == 4  # can be changed
        assert len(self.out_levels) == self.num_out
        # Smooth layers
        self.smooth1 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.toplayer = nn.Conv2d(in_channels[3], out_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(in_channels[2], out_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(in_channels[1], out_channel, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(in_channels[0], out_channel, kernel_size=1, stride=1, padding=0)
        # self.init_weights()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, features):
        assert len(features) == self.num_in

        p5 = self.toplayer(features[3])
        p4 = self._upsample_add(p5, self.latlayer1(features[2]))
        p3 = self._upsample_add(p4, self.latlayer2(features[1]))
        p2 = self._upsample_add(p3, self.latlayer3(features[0]))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        out = [p2,p3,p4,p5]
        return out[:self.num_out]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 0)
