import torch


class LinearRegression(torch.nn.Module):

    def __init__(self, input_dim=28*28, output_dim=10):
        super(LinearRegression, self).__init__()
        self.input_dim = input_dim

        # Defining the network architecture.
        self.linear = torch.nn.Linear(input_dim, output_dim)

        # Initializing the weights of the network.
        self.reset()

    def reset(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.zero_()

    def forward(self, x):
        out = x.view(-1, self.input_dim)
        return self.linear(out)
