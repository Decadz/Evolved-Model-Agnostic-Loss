import torch
import math


class PyramidNet(torch.nn.Module):

    def __init__(self, depth=110, alpha=48, num_classes=10, log_softmax=False, **kwargs):

        """
        Implementation of PyramidNet from the paper "Deep Pyramidal Residual
        Networks" by Dongyoon Han, Jiwhan Kim, and Junmo Kim
        """

        super(PyramidNet, self).__init__()

        self.inplanes = 16

        n = int((depth - 2) / 6)
        block = _BasicBlock

        self.addrate = alpha / (3 * n * 1.0)

        self.input_featuremap_dim = self.inplanes
        self.conv1 = torch.nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.input_featuremap_dim)

        self.featuremap_dim = self.input_featuremap_dim
        self.layer1 = self._make_layer(block, n)
        self.layer2 = self._make_layer(block, n, stride=2)
        self.layer3 = self._make_layer(block, n, stride=2)

        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final = torch.nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AvgPool2d(8)

        # Feed forward block
        self.fc = torch.nn.Linear(self.final_featuremap_dim, num_classes)

        # Output activation function.
        self.out = torch.nn.LogSoftmax(dim=1) if log_softmax else torch.nn.Softmax(dim=1)

        # Initializing the weights of the network.
        self.reset()

    def _make_layer(self, block, block_depth, stride=1):
        downsample = None
        if stride != 1:
            downsample = torch.nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

        layers = []
        self.featuremap_dim = self.featuremap_dim + self.addrate
        layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample))
        for i in range(1, block_depth):
            temp_featuremap_dim = self.featuremap_dim + self.addrate
            layers.append(
                block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1))
            self.featuremap_dim = temp_featuremap_dim
        self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

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
        out = self.bn1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn_final(out)
        out = self.relu_final(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return self.out(out)


class _BasicBlock(torch.nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(_BasicBlock, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn3 = torch.nn.BatchNorm2d(planes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
            featuremap_size = shortcut.size()[2:4]
        else:
            shortcut = x
            featuremap_size = out.size()[2:4]

        batch_size = out.size()[0]
        residual_channel = out.size()[1]
        shortcut_channel = shortcut.size()[1]

        if residual_channel != shortcut_channel:
            padding = torch.autograd.Variable(
                torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
                                       featuremap_size[1]).fill_(0))
            out += torch.cat((shortcut, padding), 1)
        else:
            out += shortcut

        return out


def conv3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3,
                           stride=stride, padding=1, bias=False)
