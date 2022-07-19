import torch


class AlexNet(torch.nn.Module):

    def __init__(self, output_dim=10, log_softmax=False, **kwargs):

        """
        Implementation of AlexNet from the paper "ImageNet Classification
        with Deep Convolutional Neural Networks" by Alex Krizhevsky, Ilya
        Sutskever and Geoffrey E. Hinton.
        """

        super(AlexNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 192, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0),
            torch.nn.Linear(256 * 2 * 2, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, output_dim),
        )

        # Output activation function.
        self.out = torch.nn.LogSoftmax(dim=1) if log_softmax else torch.nn.Softmax(dim=1)

        # Initializing the weights of the network.
        self.reset()

    def reset(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight)
                module.bias.data.zero_()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), 256 * 2 * 2)
        out = self.classifier(out)
        return self.out(out)
