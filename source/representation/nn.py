import torch


class NeuralNetwork(torch.nn.Module):

    def __init__(self, output_dim, reduction="mean", logits_to_prob=True, one_hot_encode=True,
                 output_activation=torch.nn.Softplus(), **kwargs):

        """
        Creating a feed forward neural network meta loss function. Code inspired
        by the paper "Meta learning via learned loss." by Bechtle et al.

        :param output_dim: Output vector dimension of base network.
        :param reduction: Reduction operator for aggregating results.
        :param logits_to_prob: Apply transform to convert predicted output to probability.
        :param one_hot_encode: Apply transform to convert label to one-hot encoded label.
        :param output_activation: Loss function output activation.
        """

        super(NeuralNetwork, self).__init__()

        # Meta-loss functions hyper-parameters.
        self.output_dim = output_dim
        self.reduction = reduction

        # Transformations to apply to the inputs.
        self.logits_to_prob = logits_to_prob
        self.one_hot_encode = one_hot_encode

        # Defining the loss functions architecture.
        self.network = torch.nn.Sequential(
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

        # Transforming the prediction and target vectors.
        y_pred, y_target = self._transform_input(y_pred, y_target)

        if self.output_dim == 1:  # If its a single-output problem.
            loss = self.network(torch.cat((y_pred, y_target), dim=1))
            return self._reduce_output(loss)

        else:  # If its a multi-output problem.
            res = []  # Iterating over each output label.
            for i in range(self.output_dim):
                yp = torch.unsqueeze(y_pred[:, i], 1)
                y = torch.unsqueeze(y_target[:, i], 1)
                res.append(self.network(torch.cat((yp, y), dim=1)))

            # Taking the mean across the classes.
            loss = torch.stack(res, dim=0).sum(axis=0)
            return self._reduce_output(loss)

    def _transform_input(self, y_pred, y_target):

        if self.logits_to_prob:  # Converting the raw logits into probabilities.
            y_pred = torch.nn.functional.sigmoid(y_pred) if self.output_dim == 1 \
                else torch.nn.functional.softmax(y_pred, dim=1)

        if self.one_hot_encode:  # If the target is not already one-hot encoded.
            y_target = torch.nn.functional.one_hot(y_target, num_classes=self.output_dim)

        return y_pred, y_target

    def _reduce_output(self, loss):
        # Applying the desired reduction operation to the loss vector.
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
