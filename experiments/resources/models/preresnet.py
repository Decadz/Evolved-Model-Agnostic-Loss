import torch
import math


class PreResNet(torch.nn.Module):

    def __init__(self, depth=110, output_dim=10, log_softmax=False, **kwargs):

        """
        Implementation of PreResNet from the paper "Identity Mappings in
        Deep Residual Networks" by Kaiming He, Xiangyu Zhang, Shaoqing
        Ren, and Jian Sun

        :param depth: Block depth, should be 6n+2, e.g. 20, 32, 44, 56, 110.
        :param output_dim: Dimensions of output vector.
        """

        super(PreResNet, self).__init__()

        n = (depth - 2) // 6
        block = _BasicBlock

        self.inplanes = 16
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.bn = torch.nn.BatchNorm2d(64 * block.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AvgPool2d(8)
        self.fc = torch.nn.Linear(64 * block.expansion, output_dim)

        # Output activation function.
        self.out = torch.nn.LogSoftmax(dim=1) if log_softmax else torch.nn.Softmax(dim=1)

        # Initializing the weights of the network.
        self.reset()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.out(out)


class _BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_BasicBlock, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


def conv3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3,
                           stride=stride, padding=1, bias=False)
