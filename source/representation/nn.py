import torch


class NeuralNetwork(torch.nn.Module):

    def __init__(self, output_dim, reduction="mean", output_activation=torch.nn.Softplus(), **kwargs):

        """
        Creating a feed forward neural network meta loss function. Code inspired
        by the paper "Meta learning via learned loss." by Bechtle et al.

        :param output_dim: Output vector dimension of base network.
        :param reduction: Reduction operator for aggregating results.
        :param output_activation: Loss function output activation.
        """

        super(NeuralNetwork, self).__init__()

        # Output dimension of the loss network.
        self.output_dim = output_dim

        # Reduction type to go from vector to scalar.
        self.reduction = reduction

        # Defining the meta-loss network.
        self.loss = torch.nn.Sequential(
            torch.nn.Linear(2, 50, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1, bias=False),
            output_activation
        )

        # Initializing the weights of the network.
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight)

    def forward(self, y_pred, y_target):

        # Single output problem (e.g. regression or binary classification).
        if self.output_dim == 1:
            loss = self._compute_loss(y_pred, y_target)

        else:  # Multi-output problem (e.g. multi-class classification).
            y_target = torch.nn.functional.one_hot(y_target, num_classes=self.output_dim)

            res = []  # Iterating over each class label in the encoded class vector.
            for i in range(len(y_pred[0])):
                yp = torch.unsqueeze(y_pred[:, i], 1)
                y = torch.unsqueeze(y_target[:, i], 1)
                res.append(self._compute_loss(yp, y))

            # Taking the summation across the classes.
            loss = torch.stack(res, dim=0).sum(axis=0)

        # Applying the reduction operation to the loss vector.
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def _compute_loss(self, y_pred, y_target):
        y = torch.cat((y_pred, y_target), dim=1)
        return self.loss(y)
