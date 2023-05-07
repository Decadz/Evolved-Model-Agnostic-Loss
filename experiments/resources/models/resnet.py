import torch


class _ResNet(torch.nn.Module):

    def __init__(self, resnet_version=50, output_dim=10, **kwargs):

        """
        Implementation of ResNets from the paper "Deep Residual Learning
        for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing
        Ren, and Jian Sun.
        """

        super(_ResNet, self).__init__()

        if resnet_version == 18:
            block, num_blocks = _BasicBlock, [2, 2, 2, 2]
        if resnet_version == 34:
            block, num_blocks = _BasicBlock, [3, 4, 6, 3]
        if resnet_version == 50:
            block, num_blocks = _Bottleneck, [3, 4, 6, 3]
        if resnet_version == 101:
            block, num_blocks = _Bottleneck, [3, 4, 23, 3]
        if resnet_version == 152:
            block, num_blocks = _Bottleneck, [3, 8, 36, 3]

        self.in_planes = 64

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = torch.nn.Linear(512 * block.expansion, output_dim)

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
                torch.nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight)
                module.bias.data.zero_()

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)


class _BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(_BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class _Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(_Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = torch.nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out


class ResNet18(_ResNet):

    def __init__(self, **kwargs):
        super(ResNet18, self).__init__(resnet_version=18, **kwargs)


class ResNet34(_ResNet):

    def __init__(self, **kwargs):
        super(ResNet34, self).__init__(resnet_version=34, **kwargs)


class ResNet50(_ResNet):

    def __init__(self, **kwargs):
        super(ResNet50, self).__init__(resnet_version=50, **kwargs)


class ResNet101(_ResNet):

    def __init__(self, **kwargs):
        super(ResNet101, self).__init__(resnet_version=101, **kwargs)


class ResNet152(_ResNet):

    def __init__(self, **kwargs):
        super(ResNet152, self).__init__(resnet_version=152, **kwargs)
