import torch
import math


class _PreResNet(torch.nn.Module):

    """
    Implementation of PreResNet from the paper "Identity Mappings in
    Deep Residual Networks" by Kaiming He, Xiangyu Zhang, Shaoqing
    Ren, and Jian Sun
    :param num_blocks: PreActivationResNet block sizes/structure.
    :param output_dim: Dimensions of output vector.
    """

    def __init__(self, num_blocks, num_classes=10, **kwargs):
        super(_PreResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(_BasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(_BasicBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(_BasicBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(_BasicBlock, 512, num_blocks[3], stride=2)
        self.avgpool = torch.nn.AvgPool2d(4)
        self.linear = torch.nn.Linear(512*_BasicBlock.expansion, num_classes)

        # Initializing the weights of the network.
        self.reset()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def reset(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight)
                module.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        out += shortcut
        return out


class _BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(_BasicBlock, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out += shortcut
        return out


class PreResNet18(_PreResNet):

    def __init__(self, **kwargs):
        super(PreResNet18, self).__init__(num_blocks=[2, 2, 2, 2], **kwargs)


class PreResNet34(_PreResNet):

    def __init__(self, **kwargs):
        super(PreResNet34, self).__init__(num_blocks=[3, 4, 6, 3], **kwargs)


class PreResNet50(_PreResNet):

    def __init__(self, **kwargs):
        super(PreResNet50, self).__init__(num_blocks=[3, 4, 6, 3], **kwargs)


class PreResNet101(_PreResNet):

    def __init__(self, **kwargs):
        super(PreResNet101, self).__init__(num_blocks=[3, 4, 23, 3], **kwargs)


class PreResNet152(_PreResNet):

    def __init__(self, **kwargs):
        super(PreResNet152, self).__init__(num_blocks=[3, 8, 36, 3], **kwargs)