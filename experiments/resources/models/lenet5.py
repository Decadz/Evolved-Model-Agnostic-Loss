import torch


class LeNet5(torch.nn.Module):

    """
    Implementation of LeNet5 from the paper "Gradient-Based Learning
    Applied to Document Recognition" by Yann LeCun, Leon Bottou, Yoshua
    Bengio and Patrick Haffner.
    """

    def __init__(self, output_dim=10, **kwargs):
        super(LeNet5, self).__init__()

        # Defining the LeNet-5 network architecture.
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.Tanh(),
            torch.nn.AvgPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(16 * 16, 120),
            torch.nn.Tanh(),
            torch.nn.Linear(120, 84),
            torch.nn.Tanh(),
            torch.nn.Linear(84, output_dim)
        )

        # Initializing the weights of the network.
        self.reset()

    def forward(self, x):
        return self.network(x)

    def reset(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.zero_()
