
# Importing the base-networks used for meta-learning.
from experiments.resources.models.linear import LinearRegression
from experiments.resources.models.mlp import MultiLayerPerceptron
from experiments.resources.models.lenet5 import LeNet5
from experiments.resources.models.alexnet import AlexNet
from experiments.resources.models.vgg import VGG11, VGG13, VGG16, VGG19
from experiments.resources.models.allcnnc import AllCNNC
from experiments.resources.models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from experiments.resources.models.preresnet import PreResNet18, PreResNet34, PreResNet50, PreResNet101, PreResNet152
from experiments.resources.models.wideresnet import WideResNet168, WideResNet404, WideResNet2810
from experiments.resources.models.squeezenet import SqueezeNet
from experiments.resources.models.pyramidnet import PyramidNet

# Importing the meta-learning datasets.
from experiments.resources.datasets import Diabetes
from experiments.resources.datasets import BostonHousing
from experiments.resources.datasets import CaliforniaHousing
from experiments.resources.datasets import MNIST
from experiments.resources.datasets import SVHN
from experiments.resources.datasets import CIFAR10
from experiments.resources.datasets import CIFAR100

# Importing utility functions for running experiments.
from experiments.resources.parser import register_configurations
from experiments.resources.metrics import MultiErrorRate
from experiments.resources.metrics import BinaryErrorRate
from experiments.resources.exporter import export_results
from experiments.resources.exporter import export_loss
from experiments.resources.exporter import export_model

import torch


class LearnedLoss(torch.nn.Module):

    def __init__(self, phi0, phi1, output_dim, reduction="mean",
                 logits_to_prob=True, one_hot_encode=True):
        super(LearnedLoss, self).__init__()

        self.output_dim = output_dim
        self.reduction = reduction

        # Transformations to apply to the inputs.
        self.logits_to_prob = logits_to_prob
        self.one_hot_encode = one_hot_encode

        self.phi0 = torch.tensor(phi0)
        self.phi1 = torch.tensor(phi1)

    def forward(self, y_pred, y_target):

        # Transforming the prediction and target vectors.
        y_pred, y_target = self._transform_input(y_pred, y_target)

        res = []  # Iterating over each output label.
        for i in range(self.output_dim):
            yp = torch.unsqueeze(y_pred[:, i], 1)
            y = torch.unsqueeze(y_target[:, i], 1)
            res.append(self.phi0 * torch.abs(torch.log(self.phi1 * (yp * y + 1e-10))))

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


class SmoothCrossEntropyLoss(torch.nn.modules.loss._WeightedLoss):

    def __init__(self, weight=None, reduction='mean', epsilon=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = epsilon
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, inputs, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, inputs.size(-1), self.smoothing)
        log_preds = torch.nn.functional.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))


dataset_archive = {
    "diabetes": {"data": Diabetes, "config": "experiments/resources/configurations/diabetes_config.yaml"},
    "boston": {"data": BostonHousing, "config": "experiments/resources/configurations/boston_config.yaml"},
    "california": {"data": CaliforniaHousing, "config": "experiments/resources/configurations/california_config.yaml"},
    "mnist": {"data": MNIST, "config": "experiments/resources/configurations/mnist_config.yaml"},
    "cifar10": {"data": CIFAR10, "config": "experiments/resources/configurations/cifar10_config.yaml"},
    "cifar100": {"data": CIFAR100, "config": "experiments/resources/configurations/cifar100_config.yaml"},
    "svhn": {"data": SVHN, "config": "experiments/resources/configurations/svhn_config.yaml"}
}


model_archive = {
    "linear": LinearRegression,
    "mlp": MultiLayerPerceptron,
    "lenet5": LeNet5,
    "alexnet": AlexNet,  # 23,272,266
    "allcnnc": AllCNNC,  # 1,372,254
    "preresnet18": PreResNet18,
    "preresnet34": PreResNet34,
    "preresnet50": PreResNet50,
    "preresnet101": PreResNet101,
    "preresnet152": PreResNet152,
    "pyramidnet": PyramidNet,
    "resnet18": ResNet18,  # 11,173,962
    "resnet34": ResNet34,  # 21,282,122
    "resnet50": ResNet50,  # 23,520,842
    "resnet101": ResNet101,  # 42,512,970
    "resnet152": ResNet152,  # 58,156,618
    "squeezenet": SqueezeNet,
    "vgg11": VGG11,  # 9,231,114
    "vgg13": VGG13,  # 9,416,010
    "vgg16": VGG16,  # 14,728,266
    "vgg19": VGG19,  # 20,040,522
    "wideresnet40-4": WideResNet404,  # 8,972,340
    "wideresnet16-8": WideResNet168,  # 11,007,540
    "wideresnet28-10": WideResNet2810  # 36,536,884
}

objective_archive = {
    "multierrorrate": MultiErrorRate(),
    "binaryerrorrate": BinaryErrorRate(),
    "nllloss": torch.nn.NLLLoss(),
    "bceloss": torch.nn.BCELoss(),
    "mseloss": torch.nn.MSELoss(),
    "celoss": torch.nn.CrossEntropyLoss(),
    "learned": LearnedLoss,
    "smoothing": SmoothCrossEntropyLoss
}
