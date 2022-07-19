import sys, os
sys.path.append(os.getcwd())

from experiments.resources import *
from source import *

import argparse
import torch
import random
import numpy
import time
import yaml

# Use the GPU/CUDA when available, else use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Getting the experiments directory for loading and saving.
directory = os.path.dirname(os.path.abspath(__file__)) + "/"

# Ensuring PyTorch gives deterministic output.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# Parsing arguments to construct experiments.
# ============================================================

parser = argparse.ArgumentParser(description="Experiment Runner")

# Loss function learning algorithm arguments.
parser.add_argument("--method", required=True, type=str)

# Experiment settings.
parser.add_argument("--source_dataset", required=True, type=str)
parser.add_argument("--source_model", required=True, type=str)
parser.add_argument("--source_seed", required=True, type=int)

parser.add_argument("--target_dataset", required=True, type=str)
parser.add_argument("--target_model", required=True, type=str)
parser.add_argument("--target_seeds", required=True, type=int, nargs="+")

parser.add_argument("--device", required=False, type=str)

# Registering all optional configuration hyper-parameters.
register_configurations(parser)

# Retrieving the dictionary of arguments.
args = parser.parse_args()

if device is not None:
    device = args.device

# ============================================================
# Available datasets and models for experiments.
# ============================================================

dataset_archive = {
    "mnist": {"data": MNIST, "config": directory + "configurations/mnist_config.yaml"},
    "cifar10": {"data": CIFAR10, "config": directory + "configurations/cifar10_config.yaml"},
    "cifar100": {"data": CIFAR100, "config": directory + "configurations/cifar100_config.yaml"},
    "svhn": {"data": SVHN, "config": directory + "configurations/svhn_config.yaml"}
}

model_archive = {
    "lenet5": LeNet5, "alexnet": AlexNet, "vgg": VGG, "allcnnc": AllCNNC,
    "resnet": ResNet, "preresnet": PreResNet, "wideresnet": WideResNet,
    "squeezenet": SqueezeNet, "pyramidnet": PyramidNet
}

# ============================================================
# Constructing and executing experiments.
# ============================================================


def _run_transfer_experiment(dataset, target_model, target_config, source_config, random_state):

    # Setting the reproducibility seed in PyTorch.
    if random_state is not None:
        torch.cuda.manual_seed(random_state)
        torch.manual_seed(random_state)
        numpy.random.seed(random_state)
        random.seed(random_state)

    # Generating the PyTorch network and dataset.
    training, validation, testing = dataset()

    # Finding the source directory to find file path to the source loss function.
    file_path = directory + source_config["output_path"] + "loss_functions/" + args.method + \
                "-" + args.source_dataset + "-" + args.source_model + "-" + str(args.source_seed) + ".pth"

    # Loading the meta-learned loss function.
    meta_loss_function = torch.load(file_path, map_location=device)
    meta_loss_function.device = device
    meta_loss_function.output_dim = target_config["base_network_parameters"]["output_dim"]

    print("Execution Seed", str(random_state), "Started")

    # Creating a dictionary for recording experiment results.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

    # Creating a loss function learning object.
    lfl = LossFunctionLearning(
        representation=None, optimization=None, base_network=target_model,
        random_state=random_state, device=device, **target_config
    )

    # Executing the training session with the meta-learned loss function.
    lfl.meta_loss_function = meta_loss_function

    # Executing the meta-testing phase with the the target domain.
    models, testing_history = lfl.meta_test(
        tasks=training, performance_metric=objective_archive[target_config["outer_objective_name"]]
    )

    # Recording the learning meta-data.
    results["test_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    results["training_inference"] = lfl.evaluate(
        models=models, tasks=training,
        performance_metric=objective_archive[target_config["outer_objective_name"]]
    )
    results["testing_inference"] = lfl.evaluate(
        models=models, tasks=testing,
        performance_metric=objective_archive[target_config["outer_objective_name"]]
    )

    results["meta-testing"] = testing_history
    results["source_seed"] = args.source_seed
    results["target_seed"] = random_state

    # Recording the experiment configurations.
    results["experiment_configuration"] = target_config.copy()

    # Defining the output results directory and file name.
    res_directory = directory + target_config["output_path"]
    file_name = args.method + "-transfer-" + args.target_dataset + "-" + args.target_model + \
                "-" + str(random_state)

    # Exporting the results to a json file.
    export_results(results, res_directory, file_name)
    export_model(models[0], res_directory, file_name)

    print("Execution Seed", str(random_state), "Complete")


objective_archive = {
    "MultiErrorRate": MultiErrorRate(), "BinaryErrorRate": BinaryErrorRate(),
    "NLLLoss": torch.nn.NLLLoss(), "BCELoss": torch.nn.BCELoss(), "MSELoss": torch.nn.MSELoss()
}

# Opening the relevant configurations file.
with open(dataset_archive[args.source_dataset]["config"]) as file:
    source_config = yaml.safe_load(file)

# Opening the relevant configurations file.
with open(dataset_archive[args.target_dataset]["config"]) as file:
    target_config = yaml.safe_load(file)

# Loading in any manually overridden hyper-parameter values.
for arg in vars(args):

    # Non empty arguments which have been manually provided.
    if getattr(args, arg) is not None and \
            arg not in {"method", "dataset", "model", "seeds", "device"}:

        # Overriding the default hyper-parameter value.
        target_config[arg] = getattr(args, arg)

# Retrieving the function for the selected dataset.
target_dataset_fn = dataset_archive[args.target_dataset]["data"]

# Retrieving the function for the selected model.
target_model_fn = model_archive[args.target_model]

# Executing the experiments using the given seeds.
for random_state in args.target_seeds:

    # Executing the experiments with the given arguments.
    _run_transfer_experiment(
        target_dataset_fn, target_model_fn, target_config,
        source_config, random_state
    )
