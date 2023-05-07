import torchvision
import torch
import os

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston
from sklearn import preprocessing

# Finding the current directory of this file.
directory = os.path.dirname(os.path.realpath(__file__)) + "/datasets"


def Diabetes():
    # Loading the dataset using sklearn.
    X, y = load_diabetes(return_X_y=True)
    return _preprocess_sklearn(X, y)


def BostonHousing():

    # Loading the dataset using sklearn.
    X, y = load_boston(return_X_y=True)
    return _preprocess_sklearn(X, y)


def CaliforniaHousing():

    # Loading the dataset using sklearn.
    X, y = fetch_california_housing(return_X_y=True)
    return _preprocess_sklearn(X, y)


def _preprocess_sklearn(X, y):

    # Partitioning the dataset into training, validation and testing sets.
    training_X, testing_X, training_y, testing_y = train_test_split(
        X, y, test_size=0.4, random_state=0)
    training_X, validation_X, training_y, validation_y = train_test_split(
        testing_X, testing_y, test_size=0.5, random_state=0)

    # Standardizing the input to improve model performance.
    scaler_X = preprocessing.StandardScaler().fit(training_X)
    training_X = scaler_X.transform(training_X)
    validation_X = scaler_X.transform(validation_X)
    testing_X = scaler_X.transform(testing_X)

    # Standardizing the output to improve model performance.
    scaler_y = preprocessing.StandardScaler().fit(training_y.reshape(-1, 1))
    training_y = scaler_y.transform(training_y.reshape(-1, 1))
    validation_y = scaler_y.transform(validation_y.reshape(-1, 1))
    testing_y = scaler_y.transform(testing_y.reshape(-1, 1))

    # Reshaping and converting to correct format.
    training_X = torch.tensor(training_X, dtype=torch.float)
    testing_X = torch.tensor(testing_X, dtype=torch.float)
    validation_X = torch.tensor(validation_X, dtype=torch.float)
    training_y = torch.tensor(training_y, dtype=torch.float)
    validation_y = torch.tensor(validation_y, dtype=torch.float)
    testing_y = torch.tensor(testing_y, dtype=torch.float)

    # Grouping the datasets into PyTorch dataset.
    training = torch.utils.data.TensorDataset(training_X, training_y)
    validation = torch.utils.data.TensorDataset(validation_X, validation_y)
    testing = torch.utils.data.TensorDataset(testing_X, testing_y)

    return training, validation, testing


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
        train_indices, val_indices, _, _ = train_test_split(
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
        train_indices, val_indices, _, _ = train_test_split(
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
        train_indices, val_indices, _, _ = train_test_split(
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
        train_indices, val_indices, _, _ = train_test_split(
            range(len(training)),
            training.labels,
            stratify=training.labels,
            test_size=0.1,
        )

        validation = torch.utils.data.Subset(training, val_indices)
        training = torch.utils.data.Subset(training, train_indices)

    return training, validation, testing

