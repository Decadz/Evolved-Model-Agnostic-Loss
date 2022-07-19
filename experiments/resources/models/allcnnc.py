import torch


class AllCNNC(torch.nn.Module):

    def __init__(self, output_dim, log_softmax=False, **kwargs):

        """
        Implementation of All-CNN-C model from the paper "Striving for
        Simplicity: The All Convolutional Net" by Jost Tobias Springenberg,
        Alexey Dosovitskiy, Thomas Brox and Martin Riedmiller.
        """

        super(AllCNNC, self).__init__()

        self.relu = torch.nn.ReLU(inplace=True)

        # First block.
        self.conv1 = torch.nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(96)
        self.conv2 = torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(96)
        self.conv3 = torch.nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(96)
        self.drop1 = torch.nn.Dropout(p=0.5)

        # Second block.
        self.conv4 = torch.nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(192)
        self.conv5 = torch.nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.bn5 = torch.nn.BatchNorm2d(192)
        self.conv6 = torch.nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(192)
        self.drop2 = torch.nn.Dropout(p=0.5)

        # Third block.
        self.conv7 = torch.nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.bn7 = torch.nn.BatchNorm2d(192)
        self.conv8 = torch.nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.bn8 = torch.nn.BatchNorm2d(192)
        self.conv9 = torch.nn.Conv2d(192, output_dim, kernel_size=1, stride=1, padding=0)
        self.bn9 = torch.nn.BatchNorm2d(output_dim)
        self.pool = torch.nn.AvgPool2d(6, output_dim)

        # Output activation function.
        self.out = torch.nn.LogSoftmax(dim=1) if log_softmax else torch.nn.Softmax(dim=1)

        # Initializing the weights of the network.
        self.reset()

    def reset(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        out = self.drop1(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))))))
        out = self.drop2(self.bn6(self.conv6(self.relu(self.bn5(self.conv5(self.relu(self.bn4(self.conv4(out)))))))))
        out = self.pool(self.bn9(self.conv9(self.relu(self.bn8(self.conv8(self.relu(self.bn7(self.conv7(out)))))))))
        out = out.view(out.size(0), -1)
        return self.out(out)
