import torch
import math


class _WideResNet(torch.nn.Module):

    def __init__(self, depth=28, widen_factor=10, output_dim=10, **kwargs):

        """
        Implementation of WideResNet from the paper "Wide Residual
        Networks" by Sergey Zagoruyko and Nikos Komodakis.
        """

        super(_WideResNet, self).__init__()

        # Defining the network block architecture.
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n = (depth - 4) // 6  # depth should be 6n+4
        block = _BasicBlock
        
        self.conv1 = torch.nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = _NetworkBlock(n, channels[0], channels[1], block, 1)
        self.block2 = _NetworkBlock(n, channels[1], channels[2], block, 2)
        self.block3 = _NetworkBlock(n, channels[2], channels[3], block, 2)
        self.bn1 = torch.nn.BatchNorm2d(channels[3])
        self.relu = torch.nn.ReLU(inplace=True)
        self.fc = torch.nn.Linear(channels[3], output_dim)
        self.nChannels = channels[3]

        # Initializing the weights of the network.
        self.reset()

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
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = torch.nn.functional.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class _BasicBlock(torch.nn.Module):

    def __init__(self, in_planes, out_planes, stride):
        super(_BasicBlock, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_planes)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and torch.nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class _NetworkBlock(torch.nn.Module):

    def __init__(self, nb_layers, in_planes, out_planes, block, stride):
        super(_NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet404(_WideResNet):

    def __init__(self, **kwargs):
        super(WideResNet404, self).__init__(depth=40, widen_factor=4, **kwargs)


class WideResNet168(_WideResNet):

    def __init__(self, **kwargs):
        super(WideResNet168, self).__init__(depth=16, widen_factor=8, **kwargs)


class WideResNet2810(_WideResNet):

    def __init__(self, **kwargs):
        super(WideResNet2810, self).__init__(depth=28, widen_factor=10, **kwargs)
