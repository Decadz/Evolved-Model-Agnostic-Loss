import torch


class SqueezeNet(torch.nn.Module):

    def __init__(self, output_dim=10, **kwargs):

        """
        Implementation of SqueezeNet from the paper "SqueezeNet: AlexNet-level
        accuracy with 50x fewer parameters and <0.5MB model size" by Forrest N.
        Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J.
        Dally and Kurt Keutzer.
        """

        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, 3, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2)
        )

        self.fire2 = _Fire(96, 128, 16)
        self.fire3 = _Fire(128, 128, 16)
        self.fire4 = _Fire(128, 256, 32)
        self.fire5 = _Fire(256, 256, 32)
        self.fire6 = _Fire(256, 384, 48)
        self.fire7 = _Fire(384, 384, 48)
        self.fire8 = _Fire(384, 512, 64)
        self.fire9 = _Fire(512, 512, 64)

        self.conv10 = torch.nn.Conv2d(512, output_dim, 1)
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.maxpool = torch.nn.MaxPool2d(2, 2)

        # Initializing the weights of the network.
        self.reset()

    def reset(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        out = self.stem(x)
        f2 = self.fire2(out)
        f3 = self.fire3(f2) + f2
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)
        f5 = self.fire5(f4) + f4
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)
        f9 = self.fire9(f8)
        c10 = self.conv10(f9)
        out = self.avg(c10)
        return out.view(out.size(0), -1)


class _Fire(torch.nn.Module):

    def __init__(self, in_channel, out_channel, squeeze_channel):
        super().__init__()
        self.squeeze = torch.nn.Sequential(
            torch.nn.Conv2d(in_channel, squeeze_channel, 1),
            torch.nn.BatchNorm2d(squeeze_channel),
            torch.nn.ReLU(inplace=True)
        )

        self.expand_1x1 = torch.nn.Sequential(
            torch.nn.Conv2d(squeeze_channel, int(out_channel / 2), 1),
            torch.nn.BatchNorm2d(int(out_channel / 2)),
            torch.nn.ReLU(inplace=True)
        )

        self.expand_3x3 = torch.nn.Sequential(
            torch.nn.Conv2d(squeeze_channel, int(out_channel / 2), 3, padding=1),
            torch.nn.BatchNorm2d(int(out_channel / 2)),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.squeeze(x)
        out = torch.cat([self.expand_1x1(out), self.expand_3x3(out)], 1)
        return out
