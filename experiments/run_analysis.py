import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import collections
import datetime
import torch
import json

# Use the GPU/CUDA when available, else use the CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    compute_results()
    #compute_run_time()
    #plot_learning_curve()
    #plot_filters()
    #plot_loss_functions()
    #plot_learning_rate()
    #plot_search_space()


def compute_results():

    paths = [
        "results/mnist/baseline-mnist-lenet5",
        "results/mnist/ml3-mnist-lenet5",
        "results/mnist/taylorglo-mnist-lenet5",
        "results/mnist/gplfl-mnist-lenet5",
        "results/mnist/evomal-mnist-lenet5",
    ]

    # Iterating over the different methods.
    for path in paths:

        training, testing = [], []
        print(path)

        # Iterating over the random seeds/executions.
        for i in range(5):

            # Loading the json file into a dictionary.
            res = json.load(open(path + "-" + str(i) + ".json"))

            # Extracting the training and testing inference from the results.
            training.append(np.mean(res["training_inference"]))
            testing.append(np.mean(res["testing_inference"]))
            #print("Seed:", str(i), res["testing_inference"])

        # Computing the mean +- std of the inference performance.
        training_mean, training_std = np.mean(training), np.std(training)
        testing_mean, testing_std = np.mean(testing), np.std(testing)

        # Displaying the results to the console.
        print("Training:", round(training_mean, 4), "$\pm$", round(training_std, 4))
        print("Testing:", round(testing_mean, 4), "$\pm$", round(testing_std, 4))
        print()


def compute_run_time():

    paths = [
        "results/mnist/ml3-mnist-lenet5",
        "results/mnist/taylorglo-mnist-lenet5",
        "results/mnist/gplfl-mnist-lenet5",
        "results/mnist/evomal-mnist-lenet5",
    ]

    # Iterating over the different methods.
    for path in paths:

        results = []
        print(path)

        # Iterating over the random seeds/executions.
        for i in range(5):

            # Loading the json file into a dictionary.
            res = json.load(open(path + "-" + str(i) + ".json"))

            # Extracting the start and end-time and computing the difference.
            start_time = datetime.datetime.fromisoformat(res["start_time"])
            end_time = datetime.datetime.fromisoformat(res["train_time"])
            difference = str(end_time - start_time).split(", ")

            # Parsing the time and converting to decimal time.
            if len(difference) == 2:
                hours = float(difference[0].split(" ")[0]) * 24
                (h, m, s) = difference[1].split(":")
                runtime = int(hours) + int(h) + round(float(m)/60, 2)
            else:
                (h, m, s) = difference[0].split(":")
                runtime = int(h) + float(m)/60

            results.append(runtime)

        # Displaying the results to the console.
        print(np.mean(results))
        print()


def plot_learning_curve():

    paths = [
        "results/mnist/taylorglo-mnist-lenet5",
        "results/mnist/gplfl-mnist-lenet5",
        "results/mnist/evomal-mnist-lenet5",
    ]

    # Setting the plot settings.
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = "x-large"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = (4.25, 5)

    # Name of the methods and their respective plotting colors.
    method_names = ["TaylorGLO", "GP-LFL", "EvoMAL"]
    color_values = ["#6cbb6c", "#eaa825", "#db4646"]

    # Iterating over the different methods.
    for path, method, color in zip(paths, method_names, color_values):
        res_method = []

        # Iterating over the random seeds/executions.
        for seed in range(5):

            # Loading the json file into a dictionary.
            results = json.load(open(path + "-" + str(seed) + ".json"))

            res = []  # Setting None values to the upper bound to stop error in percentile calculation.
            for value in results["meta-training"]:
                res.append(value if value is not None else 0.9)

            # Adding updated results to the current methods list.
            res_method.append(res)

        # Computing the average learning curve.
        y = np.mean(res_method, axis=0).flatten()
        plt.plot(range(50), y, color=color, label=method, linewidth=3)

    plt.ylabel("Error Rate")
    plt.xticks([0, 10, 20, 30, 40, 50])
    plt.grid(alpha=0.5)
    plt.tight_layout()
    #plt.legend()
    plt.show()

    # plt.savefig("meta-training-svhn-wideresnet.pdf", bbox_inches="tight")


def plot_filters():

    path = "results/mnist/evomal-mnist-lenet5"

    # Defining the plot settings.
    fig, axes = plt.subplots(nrows=5, ncols=1, constrained_layout=True)
    population_size, num_generations = 25, 50
    results = []

    # Iterating over the random seeds/executions and loading the data.
    for i in range(5):
        res = json.load(open(path + "-" + str(i) + ".json"))
        results.append([
            res["symbolic_equivalence_history"],
            res["rejection_protocol_history"],
            res["gradient_equivalence_history"]
        ])

    results = np.asarray(results)
    results = np.mean(results, axis=0).tolist()

    # Inferring the number of loss functions optimized.
    num_optimizations = np.repeat(population_size, num_generations)
    num_optimizations = (num_optimizations - np.asarray(results[0])).tolist()

    # Inferring the number of fitness evaluations.
    num_evaluations = np.repeat(population_size, num_generations)
    num_evaluations = (num_evaluations - np.asarray(results[0]) -
                       np.asarray(results[1]) - np.asarray(results[2])).tolist()

    results.append(num_optimizations)
    results.append(num_evaluations)

    # Histogram colors and sub plot titles.
    plot_colours = ["#7881c4", "#78b3c4", "#6cbb6c", "#eaa825", "#db4646"]
    plot_names = ["Symbolic Equivalence", "Rejection Protocol", "Gradient Equivalence",
                  "Loss Function Optimization", "Loss Function Evaluations"]

    def plot_histogram(frequencies, name, colour, ax):
        sns.histplot(
            range(num_generations), weights=frequencies, x=range(num_generations),
            bins=num_generations, color=colour, ax=ax
        )

        ax.set_xlim([0, num_generations])
        ax.set_ylim([0, population_size])
        ax.set_yticks([0, 10, 20])
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.title.set_text(name)
        ax.set_ylabel("")
        ax.grid(alpha=0.4)

    # Reordering the plots to be in a more logical order.
    results = [results[i] for i in [0, 3, 1, 2, 4]]
    plot_names = [plot_names[i] for i in [0, 3, 1, 2, 4]]

    # Plotting each of the histograms.
    for data, name, colour, ax in zip(results, plot_names, plot_colours, axes.flat):
        plot_histogram(data, name, colour, ax)

    plt.show()


def plot_loss_functions():

    plt.rcParams["figure.figsize"] = (18, 5)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.labelsize"] = "medium"
    colors = ["#6cbb6c", "#eaa825", "#db4646"]

    # Close to the cross-entropy loss and unsymmetrical.
    files_1 = [
        "results/cifar10/loss_functions/evomal-cifar10-squeezenet-1.pth",
        "results/cifar10/loss_functions/evomal-cifar10-wideresnet-4.pth",
        "results/cifar10/loss_functions/evomal-cifar10-resnet-1.pth",
    ]

    # Close to the hinge loss and symmetrical.
    files_2 = [
        "results/mnist/loss_functions/evomal-mnist-lenet5-0.pth",
        "results/cifar10/loss_functions/evomal-cifar10-allcnnc-2.pth",
        "results/cifar10/loss_functions/evomal-cifar10-preresnet-2.pth",
    ]

    # Unique and not like cross-entropy or hinge loss.
    files_3 = [
        "results/cifar10/loss_functions/evomal-cifar10-alexnet-3.pth",
        "results/cifar10/loss_functions/evomal-cifar10-resnet-0.pth",
        "results/cifar10/loss_functions/evomal-cifar10-squeezenet-4.pth"
    ]

    # Labels for each of the subplots.
    labels_1, labels_2, labels_3 = ["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]

    fig, axes = plt.subplots(1, 3)
    fig.add_subplot(111, frameon=False)

    # Iterating over the different sets of loss functions.
    for index, (files, labels) in enumerate(
            zip([files_1, files_2, files_3], [labels_1, labels_2, labels_3])):

        x = torch.tensor(np.linspace(0.0001, 0.9999, 1000))
        y = torch.tensor(np.ones(1000))

        # Plotting the baseline cross entropy loss, i.e. log loss.
        loss = torch.nn.BCELoss(reduction="none")
        axes[index].plot(x.cpu().data.numpy(), loss(x, y).cpu().data.numpy(),
                         linewidth=2.5, linestyle="dashed")

        # Iterating over each file in each set.
        for file, label, colour in zip(files, labels, colors):

            # Changing settings to allow easy plotting.
            loss = torch.load(file, map_location='cpu').to("cpu")
            loss.device = "cpu"
            loss.reduction = "none"
            loss.output_dim = 1

            # Equi-spaced points for plotting the loss function.
            x = torch.tensor(np.linspace(0.0001, 0.9999, 1000))
            y = torch.tensor(np.ones(1000))

            # Sampling and plotting the loss function.
            l = loss(torch.unsqueeze(y, 1).float(), torch.unsqueeze(x, 1).float())
            axes[index].plot(x.cpu().data.numpy(), l.cpu().data.numpy(),
                             label=label, linewidth=2.5, c=colour)

            print("Loss Function:", str(loss.expression_tree))

        axes[index].grid(alpha=0.5)
        axes[index].legend()

    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Predicted Probability $f_{\\theta}(x)$")
    plt.ylabel("Loss at $y=1$")
    plt.grid(False)
    plt.show()
    #plt.savefig("meta-learned-loss-functions.pdf", bbox_inches='tight', pad_inches=0)


def plot_learning_rate():

    paths = [
        "results/other/analysis/baseline-lr-cifar10-allcnnc",
        "results/other/analysis/evomal-lr-cifar10-allcnnc"
    ]

    # Setting the plot settings.
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.labelsize"] = "large"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["figure.figsize"] = (4.25, 5)

    # Set of sub-partitions used in this experiment.
    lr_values = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    label_values = ["Cross Entropy Loss", "Meta-Learned Loss"]
    color_values = ["red", "blue"]

    # Iterating over the methods we are comparing.
    for path, color, label in zip(paths, color_values, label_values):
        res_method = []

        # Iterating over all the different subset values.
        for subset in lr_values:
            res_subset = []

            # Iterating over all the random seed values.
            for seed in range(5):

                # Loading the results into memory.
                results = json.load(open(path + "-" + str(subset) + "-" + str(seed) + ".json"))
                res_subset.append(results["testing_inference"])

            # Adding the subset results to the outer method results list.
            res_method.append(res_subset)

        # Computing the confidence interval of the inference performance.
        y_avg = np.percentile(res_method, 50, axis=1).flatten()

        # Plotting the confidence interval of the inference performance.
        plt.plot(range(len(lr_values)), y_avg, marker='o', color=color, label=label)
        print(y_avg)
        for i, v in enumerate(y_avg):
            plt.text(i, v + 0.01, v, ha="center")

    # Setting up the plot configurations.
    plt.xticks(range(len(lr_values)), lr_values)
    plt.xlabel("Initial Learning Rate ($\\alpha$)")
    plt.ylabel("Error Rate")
    plt.ylim([0.05, 0.37])
    plt.tight_layout()
    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()


def plot_search_space():

    paths = [
        "results/mnist/loss_functions/evomal-mnist-lenet5",
        "results/cifar10/loss_functions/evomal-cifar10-alexnet",
        "results/cifar10/loss_functions/evomal-cifar10-vgg",
        "results/cifar10/loss_functions/evomal-cifar10-allcnnc",
        "results/cifar10/loss_functions/evomal-cifar10-resnet",
        "results/cifar10/loss_functions/evomal-cifar10-wideresnet",
        "results/cifar10/loss_functions/evomal-cifar10-squeezenet",
        "results/cifar100/loss_functions/evomal-cifar100-wideresnet",
        "results/cifar100/loss_functions/evomal-cifar100-pyramidnet",
        "results/svhn/loss_functions/evomal-svhn-wideresnet"
    ]

    # Setting the plot settings.
    plt.rcParams["figure.figsize"] = (11, 5)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.labelsize"] = "large"

    loss_function_components = []

    # Iterating over all of the meta-learned loss functions by EvoMAL.
    for file in paths:

        # Iterating over each of the independent executions of EvoMAL.
        for seed in range(5):

            # Loading the loss function into memory.
            loss = torch.load(file + "-" + str(seed) + ".pth", map_location='cpu').to("cpu")

            # Iterating over each of the nodes in the expression tree and recording its type.
            for component in loss.expression_tree:
                loss_function_components.append(component.name)

    # Computing the frequencies of occurrence.
    loss_function_components = dict(collections.Counter(loss_function_components))

    # Sorting the components into order of appearance
    loss_function_components = dict(sorted(loss_function_components.items(), key=lambda item: item[1], reverse=True))

    # Deleting y and f(x) from the dictionary.
    del loss_function_components["ARG0"]
    del loss_function_components["ARG1"]

    # Displaying the function and terminal set primitives and their respective frequencies to the console.
    for key, value in loss_function_components.items():
        print(value, key)

    # Turning key and values into lists for compatibility with matplotlib.
    names = list(loss_function_components.keys())
    values = list(loss_function_components.values())

    # Creating the bars and manually assigning the colors. # db6e6e
    colors = ["#e16464", "#e16464", "#83be83", "#e16464", "#83be83", "#e16464", "#83be83",
              "#83be83", "#83be83", "#eeba52", "#eeba52", "#83be83", "#83be83", "#83be83"]
    plt.bar(names, values, color=colors, edgecolor="k")

    # Manually creating a legend to organize the different categories.
    legend_colors = {"Binary Operator": "#e16464", "Unary Operator": "#83be83", "Random Constant": "#eeba52"}
    labels = list(legend_colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=legend_colors[label]) for label in labels]
    plt.legend(handles, labels)

    # Adding the administrative components to the
    plt.xticks(range(len(names)), names, rotation=45)
    plt.ylabel("Frequency")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
    # plt.savefig("primitive-frequencies.pdf", bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
