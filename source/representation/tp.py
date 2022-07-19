import torch


class TaylorPolynomials(torch.nn.Module):

    def __init__(self, output_dim, device, reduction="mean", output_activation=torch.nn.Softplus(), **kwargs):

        """
        Creating a cubic taylor polynomial meta loss function. Code inspired
        by the paper "Optimizing Loss Functions Through Multi-Variate Taylor
        Polynomial Parameterization." by Gonzalez et al.

        :param output_dim: Output vector dimension of base network.
        :param reduction: Reduction operator for aggregating results.
        :param output_activation: Loss function output activation.
        :param device: Device used for Pytorch related components {"cpu", "cuda"}.
        """

        super(TaylorPolynomials, self).__init__()

        self.output_dim = output_dim
        self.reduction = reduction
        self.output_activation = output_activation
        self.device = device

        for i in range(8):  # Creating and registering the parameters with the torch module.
            param = torch.nn.Parameter(torch.randn(1, requires_grad=True) * 0.001)
            self.register_parameter(name=str(i), param=param)

    def forward(self, y_pred, y_target):

        # Single output problem (e.g. regression or binary classification).
        if self.output_dim == 1:
            loss = self.output_activation(self._compute_loss(y_pred, y_target))

        else:  # Multi-output problem (e.g. multi-class classification).
            y_target = torch.nn.functional.one_hot(y_target, num_classes=self.output_dim)

            res = []  # Iterating over each class label in the encoded class vector.
            for i in range(len(y_pred[0])):
                yp = torch.unsqueeze(y_pred[:, i], 1)
                y = torch.unsqueeze(y_target[:, i], 1)
                res.append(self.output_activation(self._compute_loss(yp, y)))

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
        return self._parameters["2"] * (y_target - self._parameters["1"]) \
               + 1/2 * self._parameters["3"] * (y_target - self._parameters["1"])**2 \
               + 1/6 * self._parameters["4"] * (y_target - self._parameters["1"])**3 \
               + self._parameters["5"] * (y_pred - self._parameters["0"]) \
               * (y_target - self._parameters["1"]) + 1/2 * self._parameters["6"] \
               * (y_pred - self._parameters["0"]) * (y_target - self._parameters["1"])**2 \
               + 1/2 * self._parameters["7"] * (y_pred - self._parameters["0"])**2 \
               * (y_target - self._parameters["0"])

    def __str__(self):
        return self._parameters["2"] + "* (y -" + self._parameters["1"] + ") + 1/2 *" \
               + self._parameters["3"] + "* (y -" + self._parameters["1"] + ")**2 + 1/6 *" \
               + self._parameters["4"] + "* (y -" + self._parameters["1"] + ")**3 +" \
               + self._parameters["5"] + "* (f(x) -" + self._parameters["0"] + ") * (y -" \
               + self._parameters["1"] + ") + 1/2 *" + self._parameters["6"] + "* (f(x) -" \
               + self._parameters["0"] + ") * (y -" + self._parameters["1"] + ")**2 + 1/2 *" \
               + self._parameters["7"] + "* (f(x) -" + self._parameters["0"] + ")**2 * (y -" \
               + self._parameters["0"] + ")"
