import torch


class _VGG(torch.nn.Module):

    def __init__(self, vgg_version=16, output_dim=10, **kwargs):

        """
        Implementation of VGG from the paper "Very Deep Convolutional Networks
        for Large-Scale Image Recognition" by Karen Simonyan, Andrew Zisserman.
        """

        super(_VGG, self).__init__()

        config = {
            11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        self.features = self._make_layers(config[vgg_version])
        self.classifier = torch.nn.Linear(512, output_dim)

        # Initializing the weights of the network.
        self.reset()

    def _make_layers(self, config):
        layers = []
        in_channels = 3
        for x in config:
            if x == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [torch.nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           torch.nn.BatchNorm2d(x),
                           torch.nn.ReLU(inplace=True)]
                in_channels = x
        layers += [torch.nn.AvgPool2d(kernel_size=1, stride=1)]
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
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


class VGG11(_VGG):

    def __init__(self, **kwargs):
        super(VGG11, self).__init__(vgg_version=11, **kwargs)


class VGG13(_VGG):

    def __init__(self, **kwargs):
        super(VGG13, self).__init__(vgg_version=13, **kwargs)


class VGG16(_VGG):

    def __init__(self, **kwargs):
        super(VGG16, self).__init__(vgg_version=16, **kwargs)


class VGG19(_VGG):

    def __init__(self, **kwargs):
        super(VGG19, self).__init__(vgg_version=19, **kwargs)