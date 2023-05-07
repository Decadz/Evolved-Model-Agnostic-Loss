import torch


class TaylorPolynomials(torch.nn.Module):

    def __init__(self, output_dim, device, reduction="mean", logits_to_prob=True, one_hot_encode=True,
                 output_activation=torch.nn.Identity(), **kwargs):

        """
        Creating a cubic taylor polynomial meta loss function. Code inspired
        by the paper "Optimizing Loss Functions Through Multi-Variate Taylor
        Polynomial Parameterization." by Gonzalez et al.

        :param output_dim: Output vector dimension of base network.
        :param reduction: Reduction operator for aggregating results.
        :param logits_to_prob: Apply transform to convert predicted output to probability.
        :param one_hot_encode: Apply transform to convert label to one-hot encoded label.
        :param output_activation: Loss function output activation.
        :param device: Device used for Pytorch related components {"cpu", "cuda"}.
        """

        super(TaylorPolynomials, self).__init__()

        # Transformations to apply to the inputs.
        self.logits_to_prob = logits_to_prob
        self.one_hot_encode = one_hot_encode

        # Loss functions hyper-parameters.
        self.output_dim = output_dim
        self.reduction = reduction
        self.output_activation = output_activation
        self.device = device

        # Creating and registering the parameters with the torch module.
        self.param = torch.nn.Parameter(torch.randn(8, requires_grad=True) * 0.001)
        self.register_parameter(name=str("phi"), param=self.param)

    def forward(self, y_pred, y_target):

        # Transforming the prediction and target vectors.
        y_pred, y_target = self._transform_input(y_pred, y_target)

        # # If its a single-output problem.
        if self.output_dim == 1:
            loss = self.output_activation(self._compute_loss(y_pred, y_target))
            return self._reduce_output(loss)

        else:  # If its a multi-output problem.

            res = []  # Iterating over each class label in the encoded class vector.
            for i in range(len(y_pred[0])):
                yp = torch.unsqueeze(y_pred[:, i], 1)
                y = torch.unsqueeze(y_target[:, i], 1)
                res.append(self.output_activation(self._compute_loss(yp, y)))

            # Taking the summation across the classes.
            loss = torch.stack(res, dim=0).sum(axis=0)
            return self._reduce_output(loss)

    def _compute_loss(self, y_pred, y_target):
        return self.param[2] * (y_target - self.param[1]) \
               + 1/2 * self.param[3] * (y_target - self.param[1])**2 \
               + 1/6 * self.param[4] * (y_target - self.param[1])**3 \
               + self.param[5] * (y_pred - self.param[0]) \
               * (y_target - self.param[1]) + 1/2 * self.param[6] \
               * (y_pred - self.param[0]) * (y_target - self.param[1])**2 \
               + 1/2 * self.param[7] * (y_pred - self.param[0])**2 \
               * (y_target - self.param[0])

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

    def __str__(self):
        return self.param["2"] + "* (y -" + self.param["1"] + ") + 1/2 *" \
               + self.param["3"] + "* (y -" + self.param["1"] + ")**2 + 1/6 *" \
               + self.param["4"] + "* (y -" + self.param["1"] + ")**3 +" \
               + self.param["5"] + "* (f(x) -" + self.param["0"] + ") * (y -" \
               + self.param["1"] + ") + 1/2 *" + self.param["6"] + "* (f(x) -" \
               + self.param["0"] + ") * (y -" + self.param["1"] + ")**2 + 1/2 *" \
               + self.param["7"] + "* (f(x) -" + self.param["0"] + ")**2 * (y -" \
               + self.param["0"] + ")"
