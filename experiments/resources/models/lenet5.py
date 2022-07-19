import torch


class LeNet5(torch.nn.Module):

    def __init__(self, channel_dim=1, num_conv_1=6, num_conv_2=16,
                 hidden_dim_1=120, hidden_dim_2=84, output_dim=10,
                 log_softmax=False, **kwargs):

        """
        Implementation of LeNet5 from the paper "Gradient-Based Learning
        Applied to Document Recognition" by Yann LeCun, Leon Bottou, Yoshua
        Bengio and Patrick Haffner.

        :param channel_dim: Number of dimensions of the input image.
        :param num_conv_1: Number of convolution filters in the first conv layer.
        :param num_conv_2: Number of convolution filters in the second conv layer.
        :param hidden_dim_1: Dimensions of hidden layer 1.
        :param hidden_dim_2: Dimensions of hidden layer 2.
        :param output_dim: Dimensions of output vector.
        """

        super(LeNet5, self).__init__()

        # Convolutional block 1.
        self.conv1 = torch.nn.Conv2d(in_channels=channel_dim, out_channels=num_conv_1, kernel_size=5)
        self.tanh1 = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2)

        # Convolutional block 2.
        self.conv2 = torch.nn.Conv2d(in_channels=num_conv_1, out_channels=num_conv_2, kernel_size=5)
        self.tanh2 = torch.nn.Tanh()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2)

        # Feed-forward block.
        self.fc1 = torch.nn.Linear(16 * num_conv_2, hidden_dim_1)
        self.tanh3 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(hidden_dim_1, hidden_dim_2)
        self.tanh4 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(hidden_dim_2, output_dim)

        # Output activation function.
        self.out = torch.nn.LogSoftmax(dim=1) if log_softmax else torch.nn.Softmax(dim=1)

        # Initializing the weights of the network.
        self.reset()

    def reset(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.zero_()

    def forward(self, x):
        out = self.pool1(self.tanh1(self.conv1(x)))
        out = self.pool2(self.tanh2(self.conv2(out)))
        out = torch.flatten(out, 1)
        out = self.tanh3(self.fc1(out))
        out = self.tanh4(self.fc2(out))
        return self.out(self.fc3(out))
