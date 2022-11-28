import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Type
from GLRA import GLRA
from ACMLP import ACMLP

def autopad(kernel, padding=None):
    # PAD TO 'SAME'
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding

class Conv(nn.Module):
    # STANDARD CONVOLUTION
    def __init__(self, input_channels, output_channels, kernel, stride, activation_type: Type[nn.Module],
                 padding: int = None, groups: int = None):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, output_channels, kernel, stride, autopad(kernel, padding),
                              groups=groups or 1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.act = activation_type()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class GroupedConvBlock(nn.Module):
    """
    Grouped Conv KxK -> usual Conv 1x1
    """

    def __init__(self, input_channels, output_channels, kernel, stride, activation_type: Type[nn.Module],
                 padding: int = None, groups: int = None):
        """
        :param groups:  num of groups in the first conv; if None depthwise separable conv will be used
                        (groups = input channels)
        """
        super().__init__()

        self.dconv = Conv(input_channels, input_channels, kernel, stride, activation_type, padding,
                          groups=groups or input_channels)
        self.conv = Conv(input_channels, output_channels, 1, 1, activation_type)

    def forward(self, x):
        return self.conv(self.dconv(x))


class Bottleneck(nn.Module):
    # STANDARD BOTTLENECK
    def __init__(self, input_channels, output_channels, shortcut: bool, activation_type: Type[nn.Module],
                 depthwise=False):
        super().__init__()

        ConvBlock = GroupedConvBlock if depthwise else Conv
        hidden_channels = output_channels
        self.cv1 = Conv(input_channels, hidden_channels, 1, 1, activation_type)
        self.cv2 = ConvBlock(hidden_channels, output_channels, 3, 1, activation_type)
        self.add = shortcut and input_channels == output_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class CSP_Bottleneck(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/detection_models/csp_darknet53.py#L148
    
    def __init__(self, input_channels, output_channels, bottleneck_blocks_num, activation_type: Type[nn.Module],
                 shortcut=True, depthwise=False, expansion=0.5):
        super().__init__()

        if output_channels%2==0:
            hidden_channels = int(output_channels * expansion)
            hidden_channels_evn = hidden_channels
        else:
            hidden_channels = int(output_channels * expansion)
            hidden_channels_evn = hidden_channels+1
        # hidden_channels = int(output_channels * expansion)
        # hidden_channels_evn = hidden_channels+1

        self.cv1 = Conv(input_channels, hidden_channels_evn, 1, 1, activation_type)
        self.cv2 = nn.Conv2d(input_channels, hidden_channels, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(hidden_channels_evn, hidden_channels_evn, 1, 1, bias=False)
        self.cv4 = Conv(hidden_channels+hidden_channels_evn, output_channels, 1, 1, activation_type)
        self.bn = nn.BatchNorm2d(hidden_channels+hidden_channels_evn)  # APPLIED TO CAT(CV2, CV3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(hidden_channels_evn, hidden_channels_evn, shortcut, activation_type, depthwise)
                                 for _ in range(bottleneck_blocks_num)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        out_2 = self.act(self.bn(torch.cat((y1, y2), dim=1)))
        out_1 = self.cv4(out_2)
        return out_1

class CSP_GLRA(nn.Module):
    def __init__(self, input_channels, output_channels, bottleneck_blocks_num, activation_type: Type[nn.Module],
                 shortcut=True, depthwise=False, expansion=0.5):
        super().__init__()

        if output_channels%2==0:
            hidden_channels = int(output_channels * expansion)
            hidden_channels_evn = hidden_channels
        else:
            hidden_channels = int(output_channels * expansion)+1
            hidden_channels_evn = hidden_channels

        self.cv1 = Conv(input_channels, hidden_channels_evn, 1, 1, activation_type)
        self.cv2 = nn.Conv2d(input_channels, hidden_channels, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(hidden_channels_evn, hidden_channels_evn, 1, 1, bias=False)
        self.cv4 = Conv(hidden_channels+hidden_channels_evn, output_channels, 1, 1, activation_type)
        self.bn = nn.BatchNorm2d(hidden_channels+hidden_channels_evn)  # APPLIED TO CAT(CV2, CV3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[GLRA(hidden_channels_evn) for _ in range(bottleneck_blocks_num)])


    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))

        y2 = self.cv2(x)
        out_2 = self.act(self.bn(torch.cat((y1, y2), dim=1)))
        out_1 = self.cv4(out_2)
        return out_1

class CSP_ACMLP(nn.Module):
    def __init__(self, input_channels, output_channels, bottleneck_blocks_num, activation_type: Type[nn.Module],
                 shortcut=True, depthwise=False, expansion=0.5):
        super().__init__()
        
        if output_channels%2==0:
            hidden_channels = int(output_channels * expansion)
            hidden_channels_evn = hidden_channels
        else:
            hidden_channels = int(output_channels * expansion)+1
            hidden_channels_evn = hidden_channels

        self.cv1 = Conv(input_channels, hidden_channels_evn, 1, 1, activation_type)
        self.cv2 = nn.Conv2d(input_channels, hidden_channels, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(hidden_channels_evn, hidden_channels_evn, 1, 1, bias=False)
        self.cv4 = Conv(hidden_channels+hidden_channels_evn, output_channels, 1, 1, activation_type)
        self.bn = nn.BatchNorm2d(hidden_channels+hidden_channels_evn)  # APPLIED TO CAT(CV2, CV3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[ACMLP(hidden_channels_evn, M =3, r=2, L=32, MLP=False) 
                                for _ in range(bottleneck_blocks_num)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))

        y2 = self.cv2(x)
        out_2 = self.act(self.bn(torch.cat((y1, y2), dim=1)))
        out_1 = self.cv4(out_2)
        return out_1
