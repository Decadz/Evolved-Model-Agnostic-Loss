import torch


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim=28*28, hidden_dim=1000, output_dim=10):
        super(MultiLayerPerceptron, self).__init__()
        self.input_dim = input_dim

        # Defining the network architecture.
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)

        # Initializing the weights of the network.
        self.reset()

    def reset(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.zero_()

    def forward(self, x):
        out = x.view(-1, self.input_dim)
        out = torch.nn.functional.relu(self.linear1(out))
        out = torch.nn.functional.relu(self.linear2(out))
        return self.linear3(out)
