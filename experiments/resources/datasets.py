import sklearn.model_selection
import torchvision
import torch
import os

# Finding the current directory of this file.
directory = os.path.dirname(os.path.realpath(__file__)) + "/datasets"


def MNIST(validation_set=True):

    # Loading the training set.
    training = torchvision.datasets.MNIST(
        train=True, root=directory, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    )

    # Loading the testing set.
    testing = torchvision.datasets.MNIST(
        train=False, root=directory, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    )

    validation = None

    if validation_set is True:  # If we want to create a validation set.

        # Partitioning the data into training and validation sets.
        train_indices, val_indices, _, _ = sklearn.model_selection.train_test_split(
            range(len(training)),
            training.targets,
            stratify=training.targets,
            test_size=0.1,
        )

        validation = torch.utils.data.Subset(training, val_indices)
        training = torch.utils.data.Subset(training, train_indices)

    return training, validation, testing


def CIFAR10(validation_set=True):

    # Loading the CIFAR-10 training and testing sets.
    training = torchvision.datasets.CIFAR10(
        train=True, root=directory, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    )

    testing = torchvision.datasets.CIFAR10(
        train=False, root=directory, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    )

    validation = None

    if validation_set is True:  # If we want to create a validation set.

        # Partitioning the data into training and validation sets.
        train_indices, val_indices, _, _ = sklearn.model_selection.train_test_split(
            range(len(training)),
            training.targets,
            stratify=training.targets,
            test_size=0.1,
        )

        validation = torch.utils.data.Subset(training, val_indices)
        training = torch.utils.data.Subset(training, train_indices)

    return training, validation, testing


def CIFAR100(validation_set=True):

    # Loading the CIFAR-100 training and testing sets.
    training = torchvision.datasets.CIFAR100(
        train=True, root=directory, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    )

    testing = torchvision.datasets.CIFAR100(
        train=False, root=directory, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    )

    validation = None

    if validation_set is True:  # If we want to create a validation set.

        # Partitioning the data into training and validation sets.
        train_indices, val_indices, _, _ = sklearn.model_selection.train_test_split(
            range(len(training)),
            training.targets,
            stratify=training.targets,
            test_size=0.1,
        )

        validation = torch.utils.data.Subset(training, val_indices)
        training = torch.utils.data.Subset(training, train_indices)

    return training, validation, testing


def SVHN(validation_set=True):

    # Loading the SVHN training and testing sets.
    training = torchvision.datasets.SVHN(
        split="train", root=directory, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744))
        ])
    )

    testing = torchvision.datasets.SVHN(
        split="test", root=directory, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4380, 0.4440, 0.4730), (0.1751, 0.1771, 0.1744))
        ])
    )

    validation = None

    if validation_set is True:  # If we want to create a validation set.

        # Partitioning the data into training and validation sets.
        train_indices, val_indices, _, _ = sklearn.model_selection.train_test_split(
            range(len(training)),
            training.labels,
            stratify=training.labels,
            test_size=0.1,
        )

        validation = torch.utils.data.Subset(training, val_indices)
        training = torch.utils.data.Subset(training, train_indices)

    return training, validation, testing
