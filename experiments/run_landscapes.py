import sys, os
sys.path.append(os.getcwd())

from experiments.resources import *

import pyvista as pv
import loss_landscapes
import numpy as np
import argparse
import torch
import os

# Getting the experiments directory for loading and saving.
directory = os.path.dirname(os.path.abspath(__file__)) + "/"

# Ensuring PyTorch gives deterministic output.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# Parsing arguments to construct experiments.
# ============================================================

parser = argparse.ArgumentParser(description="Experiment Runner")

# Experiment settings.
parser.add_argument("--method", required=True, type=str)
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--distance", required=True, type=float)
parser.add_argument("--steps", required=True, type=int)
parser.add_argument("--seeds", required=True, type=int, nargs="+")

# Retrieving the dictionary of arguments.
args = parser.parse_args()

# Defining the input and output directories.
model_directory = "experiments/results/" + args.dataset + "/models/"
loss_directory = "experiments/results/" + args.dataset + "/loss_functions/"
out_directory = "experiments/results/" + args.dataset + "/surfaces/"


def _load_model(path, method_name, dataset_name, model_name, seed):
    file = path + method_name + "-" + dataset_name + "-" + model_name + "-" + str(seed) + ".pth"
    return torch.load(file, map_location="cpu").to("cpu")


def _load_loss(path, method_name, dataset_name, model_name, seed):
    if method_name != "baseline":
        file = path + method_name + "-" + dataset_name + "-" + model_name + "-" + str(seed) + ".pth"
        loss_function = torch.load(file, map_location='cpu').to("cpu")
        loss_function.device = "cpu"
    else:
        loss_function = torch.nn.NLLLoss()
    return loss_function


def _load_dataset(dataset_name):
    if dataset_name == "mnist":
        dataset, _, _ = MNIST()
    elif dataset_name == "cifar10":
        dataset, _, _ = CIFAR10()
    elif dataset_name == "cifar100":
        dataset, _, _ = CIFAR100()
    elif dataset_name == "svhn":
        dataset, _, _ = SVHN()
    return dataset


def _visualize_loss_landscape(model, loss_function, dataset, out_directory, method_name,
                              dataset_name, model_name, seed, distance=1, steps=200,
                              smoothing_iterations=1000, show=False):

    # Creating a PyTorch DataLoader for loading instances in batches.
    generator = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    x, y = iter(generator).__next__()

    # Creating required objects for generating the loss landscape visualization.
    metric = loss_landscapes.metrics.Loss(loss_function, x, y)
    Z = loss_landscapes.random_plane(model, metric, normalization="filter", distance=distance, steps=steps)
    X, Y = np.meshgrid(np.arange(0, steps, 1), np.arange(0, steps, 1))

    # Normalizing the points to enable comparisons.
    X = (np.expand_dims(X, axis=2) - min(X.ravel()))/(max(X.ravel()) - min(X.ravel()))
    Y = (np.expand_dims(Y, axis=2) - min(Y.ravel()))/(max(Y.ravel()) - min(Y.ravel()))
    Z = (np.expand_dims(Z, axis=2) - min(Z.ravel()))/(max(Z.ravel()) - min(Z.ravel()))

    # Generating a PyVista object for representing the loss surface.
    surface = pv.StructuredGrid(X, Y, Z)
    surface["scalars"] = Z.ravel(order="f")
    surface = surface.extract_geometry().smooth(smoothing_iterations)

    if show:  # If desired the plot is displayed in a pop up window
        surface.plot(smooth_shading=True, show_edges=False, show_grid=False,
                     opacity=1, background="white", cmap="inferno")

    # Making the output directory if it doesnt exist.
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # Saving a .vtk file which can be opened in ParaView.
    surface.save(out_directory + method_name + "-" + dataset_name + "-" + model_name
                 + "-" + str(seed) + "-" + str(distance) + "-" + str(steps) + ".vtk")


# Executing the experiments with the given arguments.
for random_state in args.seeds:

    # Loading the required data to generate the loss surface visualization.
    model = _load_model(model_directory, args.method, args.dataset, args.model, random_state)
    loss_function = _load_loss(loss_directory, args.method, args.dataset, args.model, random_state)
    dataset = _load_dataset(args.dataset)

    # Visualizing the loss surface and saving a output file to the output directory.
    _visualize_loss_landscape(model, loss_function, dataset, out_directory, args.method,
                              args.dataset, args.model, random_state,
                              distance=args.distance, steps=args.steps)

    print("Finished:", out_directory + args.method + "-" + args.dataset + "-" + args.model
          + "-" + str(random_state) + "-" + str(args.distance) + "-" + str(args.steps) + ".vtk")
