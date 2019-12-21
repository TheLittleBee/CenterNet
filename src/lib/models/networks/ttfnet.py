import torch.nn as nn
import torch.nn.functional as F

from .fpn import fill_fc_weights


class TTFHead(nn.Module):
    def __init__(self, planes, num_classes, head_conv=64):
        super().__init__()

        self.planes = planes
        self.head_conv = head_conv
        self.num_classes = num_classes

        self.hm = nn.Sequential(
            nn.Conv2d(planes, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        self.wh = nn.Sequential(
            nn.Conv2d(planes, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 4,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.hm)
        fill_fc_weights(self.wh)
        self.hm[-1].bias.data.fill_(-2.19)
        self.wh[-1].bias.data.fill_(0.5)

    def forward(self, x):
        hm = self.hm(x)
        wh = F.relu(self.wh(x))

        return hm, wh
