import sys, os
sys.path.append(os.getcwd())

from experiments.resources import *
from source import *

import functools
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

# Experiment settings.
parser.add_argument("--method", required=True, type=str)
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--dataset", required=True, type=str)

parser.add_argument("--learning_rate", required=True, type=float)

parser.add_argument("--seeds", required=True, type=int, nargs="+")
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


def _run_learning_rate_experiments(dataset, model, config, random_state):

    # Setting the reproducibility seed in PyTorch.
    if random_state is not None:
        torch.cuda.manual_seed(random_state)
        torch.manual_seed(random_state)
        numpy.random.seed(random_state)
        random.seed(random_state)

    # Generating the PyTorch network and dataset.
    training, validation, testing = dataset()

    print("Execution Seed", str(random_state), "Started")

    # Creating a dictionary for recording experiment results.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

    if args.method == "baseline":

        # Using the task loss as the loss function.
        meta_loss_function = objective_archive[config["inner_objective_name"]]
        model = functools.partial(model, log_softmax=True)

        # Creating a loss function learning object.
        lfl = LossFunctionLearning(
            representation=None, optimization=None, base_network=model,
            random_state=random_state, device=device, **config
        )

        # Executing the training session with the meta-learned loss function and the desired learning rate.
        lfl.meta_loss_function = meta_loss_function
        lfl.testing_learning_rate = args.learning_rate

        # Executing the meta-testing phase with the the target domain.
        models, testing_history = lfl.meta_test(
            tasks=training, performance_metric=objective_archive[config["outer_objective_name"]]
        )

    else:
        gimli = GeneralizedInnerLoopMetaLearning(
            base_network=model, random_state=random_state, device=device,
            performance_metric=objective_archive[config["outer_objective_name"]],
            task_loss=objective_archive[config["inner_objective_name"]], **config
        )
        gimli.verbose = 0

        ea = EvolutionaryAlgorithm(
            base_network=model, random_state=random_state, device=device,
            performance_metric=objective_archive[config["outer_objective_name"]],
            task_loss=objective_archive[config["inner_objective_name"]],
            local_search=gimli, **config
        )

        lfl = LossFunctionLearning(
            representation="gp", optimization=ea, base_network=model,
            random_state=random_state, device=device, **config
        )

        lfl.testing_learning_rate = args.learning_rate
        lfl.base_learning_rate = args.learning_rate

        # Executing the meta-training phase of the algorithm.
        loss_function, training_history = lfl.meta_train(training, validation)
        results["meta-training"] = training_history

        models, testing_history = lfl.meta_test(
            tasks=training, performance_metric=objective_archive[config["outer_objective_name"]]
        )

    # Recording the learning meta-data.
    results["test_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    results["training_inference"] = lfl.evaluate(
        models=models, tasks=training,
        performance_metric=objective_archive[config["outer_objective_name"]]
    )
    results["testing_inference"] = lfl.evaluate(
        models=models, tasks=testing,
        performance_metric=objective_archive[config["outer_objective_name"]]
    )

    results["meta-testing"] = testing_history

    # Recording the experiment configurations.
    results["experiment_configuration"] = config.copy()

    # Defining the output results directory and file name.
    res_directory = directory + config["output_path"]
    file_name = args.method + "-lr-" + args.dataset + "-" + args.model + \
                "-" + str(args.learning_rate) + "-" + str(random_state)

    # Exporting the results to a json file.
    export_results(results, res_directory, file_name)

    print("Execution Seed", str(random_state), "Complete")


objective_archive = {
    "MultiErrorRate": MultiErrorRate(), "BinaryErrorRate": BinaryErrorRate(),
    "NLLLoss": torch.nn.NLLLoss(), "BCELoss": torch.nn.BCELoss(), "MSELoss": torch.nn.MSELoss()
}

# Opening the relevant configurations file.
with open(dataset_archive[args.dataset]["config"]) as file:
    config = yaml.safe_load(file)

# Loading in any manually overridden hyper-parameter values.
for arg in vars(args):

    # Non empty arguments which have been manually provided.
    if getattr(args, arg) is not None and \
            arg not in {"method", "dataset", "model", "seeds", "device"}:

        # Overriding the default hyper-parameter value.
        config[arg] = getattr(args, arg)

# Retrieving the function for the selected dataset.
dataset_fn = dataset_archive[args.dataset]["data"]

# Retrieving the function for the selected model.
model_fn = model_archive[args.model]

# Executing the experiments using the given seeds.
for random_state in args.seeds:

    # Executing the experiments with the given arguments.
    _run_learning_rate_experiments(
        dataset_fn, model_fn, config, random_state
    )
