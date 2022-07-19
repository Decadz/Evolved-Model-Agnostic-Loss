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

# Loss function learning algorithm arguments.
parser.add_argument("--method", required=True, type=str)

# Experiment settings.
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--model", required=True, type=str)
parser.add_argument("--seeds", required=True, type=int, nargs="+")
parser.add_argument("--device", required=False, type=str)

# Registering all optional configuration hyper-parameters.
register_configurations(parser)

# Retrieving the dictionary of arguments.
args = parser.parse_args()

if args.device is not None:
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


def _run_baseline_experiment(dataset, model, config, random_state):

    # Setting the reproducibility seed in PyTorch.
    if random_state is not None:
        torch.cuda.manual_seed(random_state)
        torch.manual_seed(random_state)
        numpy.random.seed(random_state)
        random.seed(random_state)

    # Generating the PyTorch network and dataset.
    training, validation, testing = dataset()

    # Defining the output results directory and file name.
    res_directory = directory + config["output_path"]
    file_name = "baseline-" + args.dataset + "-" + args.model + "-" + str(random_state)

    print(args.method, args.dataset, args.model, "seed", str(random_state), "started")

    # Creating a dictionary for recording experiment results.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

    lfl = LossFunctionLearning(
        representation=None, optimization=None,
        base_network=functools.partial(model, log_softmax=True),
        random_state=random_state, device=device, **config
    )

    # Executing the training session with the task loss.
    lfl.meta_loss_function = objective_archive[config["inner_objective_name"]]

    # Evaluating the loss function.
    if isinstance(training, list) or isinstance(testing, list):
        _evaluate_meta_dataset(lfl, training, testing, results, res_directory, file_name)
    else:
        _evaluate_standard_dataset(lfl, training, testing, results, res_directory, file_name)


def _run_gplfl_experiment(dataset, model, config, random_state):

    # Setting the reproducibility seed in PyTorch.
    if random_state is not None:
        torch.cuda.manual_seed(random_state)
        torch.manual_seed(random_state)
        numpy.random.seed(random_state)
        random.seed(random_state)

    # Generating the PyTorch network and dataset.
    training, validation, testing = dataset()

    # Defining the output results directory and file name.
    res_directory = directory + config["output_path"]
    file_name = "gplfl-" + args.dataset + "-" + args.model + "-" + str(random_state)

    print(args.method, args.dataset, args.model, "seed", str(random_state), "started")

    # Creating a dictionary for recording experiment results.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

    ea = EvolutionaryAlgorithm(
        base_network=model, random_state=random_state, device=device,
        performance_metric=objective_archive[config["outer_objective_name"]],
        task_loss=objective_archive[config["inner_objective_name"]],
        local_search=None, **config
    )

    lfl = LossFunctionLearning(
        representation="gp", optimization=ea, base_network=model,
        random_state=random_state, device=device, **config
    )

    # Executing the meta-training phase of the algorithm.
    loss_function, training_history = lfl.meta_train(training, validation)
    results["train_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Exporting the meta-learned loss function.
    export_loss(lfl.meta_loss_function, res_directory, file_name)
    results.update(training_history)

    # Evaluating the meta-learned loss function.
    if isinstance(training, list) or isinstance(testing, list):
        _evaluate_meta_dataset(lfl, training, testing, results, res_directory, file_name)
    else:
        _evaluate_standard_dataset(lfl, training, testing, results, res_directory, file_name)


def _run_ml3_experiment(dataset, model, config, random_state):

    # Setting the reproducibility seed in PyTorch.
    if random_state is not None:
        torch.cuda.manual_seed(random_state)
        torch.manual_seed(random_state)
        numpy.random.seed(random_state)
        random.seed(random_state)

    # Generating the PyTorch network and dataset.
    training, validation, testing = dataset()

    # Defining the output results directory and file name.
    res_directory = directory + config["output_path"]
    file_name = "ml3-" + args.dataset + "-" + args.model + "-" + str(random_state)

    print(args.method, args.dataset, args.model, "seed", str(random_state), "started")

    # Creating a dictionary for recording experiment results.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

    gimli = GeneralizedInnerLoopMetaLearning(
        base_network=model, random_state=random_state, device=device,
        performance_metric=objective_archive[config["outer_objective_name"]],
        task_loss=objective_archive[config["inner_objective_name"]],
        **config
    )

    lfl = LossFunctionLearning(
        representation="nn", optimization=gimli, base_network=model,
        random_state=random_state, device=device, **config
    )

    # Executing the meta-training phase of the algorithm.
    loss_function, training_history = lfl.meta_train(training, validation)
    results["train_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Exporting the meta-learned loss function.
    export_loss(lfl.meta_loss_function, res_directory, file_name)
    results.update(training_history)

    # Evaluating the meta-learned loss function.
    if isinstance(training, list) or isinstance(testing, list):
        _evaluate_meta_dataset(lfl, training, testing, results, res_directory, file_name)
    else:
        _evaluate_standard_dataset(lfl, training, testing, results, res_directory, file_name)


def _run_taylorglo_experiment(dataset, model, config, random_state):

    # Setting the reproducibility seed in PyTorch.
    if random_state is not None:
        torch.cuda.manual_seed(random_state)
        torch.manual_seed(random_state)
        numpy.random.seed(random_state)
        random.seed(random_state)

    # Generating the PyTorch network and dataset.
    training, validation, testing = dataset()

    # Defining the output results directory and file name.
    res_directory = directory + config["output_path"]
    file_name = "taylorglo-" + args.dataset + "-" + args.model + "-" + str(random_state)

    print(args.method, args.dataset, args.model, "seed", str(random_state), "started")

    # Creating a dictionary for recording experiment results.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

    cmaes = CovarianceMatrixAdaptation(
        base_network=model, random_state=random_state, device=device,
        performance_metric=objective_archive[config["outer_objective_name"]],
        task_loss=objective_archive[config["inner_objective_name"]],
        local_search=None, **config
    )

    lfl = LossFunctionLearning(
        representation="tp", optimization=cmaes, base_network=model,
        random_state=random_state, device=device, **config
    )

    # Executing the meta-training phase of the algorithm.
    loss_function, training_history = lfl.meta_train(training, validation)
    results["train_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Exporting the meta-learned loss function.
    export_loss(lfl.meta_loss_function, res_directory, file_name)
    results.update(training_history)

    # Evaluating the meta-learned loss function.
    if isinstance(training, list) or isinstance(testing, list):
        _evaluate_meta_dataset(lfl, training, testing, results, res_directory, file_name)
    else:
        _evaluate_standard_dataset(lfl, training, testing, results, res_directory, file_name)


def _run_evomal_experiment(dataset, model, config, random_state):

    # Setting the reproducibility seed in PyTorch.
    if random_state is not None:
        torch.cuda.manual_seed(random_state)
        torch.manual_seed(random_state)
        numpy.random.seed(random_state)
        random.seed(random_state)

    # Generating the PyTorch network and dataset.
    training, validation, testing = dataset()

    # Defining the output results directory and file name.
    res_directory = directory + config["output_path"]
    file_name = "evomal-" + args.dataset + "-" + args.model + "-" + str(random_state)

    print(args.method, args.dataset, args.model, "seed", str(random_state), "started")

    # Creating a dictionary for recording experiment results.
    results = {"start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}

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

    # Executing the meta-training phase of the algorithm.
    loss_function, training_history = lfl.meta_train(training, validation)
    results["train_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Exporting the meta-learned loss function.
    export_loss(lfl.meta_loss_function, res_directory, file_name)
    results.update(training_history)

    # Evaluating the meta-learned loss function.
    if isinstance(training, list) or isinstance(testing, list):
        _evaluate_meta_dataset(lfl, training, testing, results, res_directory, file_name)
    else:
        _evaluate_standard_dataset(lfl, training, testing, results, res_directory, file_name)


def _evaluate_standard_dataset(lfl, training, testing, results, res_directory, file_name):

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

    # Exporting the results to a json file.
    export_results(results, res_directory, file_name)
    export_model(models[0], res_directory, file_name)

    print(args.method, args.dataset, args.model, "seed", str(random_state), "complete")


def _evaluate_meta_dataset(lfl, training, testing, results, res_directory, file_name):

    training_models, training_history = lfl.meta_test(
        tasks=training, performance_metric=objective_archive[config["outer_objective_name"]]
    )

    # Recording the learning meta-data.
    results["test_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    results["training_inference"] = lfl.evaluate(
        models=training_models, tasks=training,
        performance_metric=objective_archive[config["outer_objective_name"]]
    )

    testing_models, testing_history = lfl.meta_test(
        tasks=testing, performance_metric=objective_archive[config["outer_objective_name"]]
    )
    results["testing_inference"] = lfl.evaluate(
        models=testing_models, tasks=testing,
        performance_metric=objective_archive[config["outer_objective_name"]]
    )

    results["meta-testing-in-sample"] = training_history
    results["meta-testing-out-sample"] = testing_history

    # Recording the experiment configurations.
    results["experiment_configuration"] = config.copy()

    # Exporting the results to a json file.
    export_results(results, res_directory, file_name)
    print(args.method, args.dataset, args.model, "seed", str(random_state), "complete")


experiment_executors = {
    "baseline": _run_baseline_experiment,
    "gplfl": _run_gplfl_experiment,
    "ml3": _run_ml3_experiment,
    "taylorglo": _run_taylorglo_experiment,
    "evomal": _run_evomal_experiment
}

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

# Executing the experiments with the given arguments.
for random_state in args.seeds:
    experiment_executors[args.method](dataset_fn, model_fn, config, random_state)
