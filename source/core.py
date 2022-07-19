from source.optimization import bp, inference
from source.representation import nn, tp

import random
import numpy
import torch
import os


class LossFunctionLearning:

    def __init__(self, representation, optimization, base_network, base_network_parameters,
                 testing_gradient_steps, testing_learning_rate, testing_weight_decay,
                 testing_momentum, testing_nesterov, testing_batch_size, testing_milestones,
                 testing_gamma, device, random_state, verbose=0, **kwargs):

        """
        Loss function learning object which is used to manage the meta-learning of
        performant loss functions. Offers a standardized interface for running loss
        function learning experiments.

        :param representation: Loss function learning representation {gp, nn, tp}.
        :param optimization: Loss function learning optimizer object.
        :param testing_gradient_steps: Number of meta-testing gradient steps.
        :param testing_learning_rate: Meta-testing learning rate.
        :param testing_weight_decay: Meta-testing weight decay amount.
        :param testing_momentum: Meta-testing momentum amount.
        :param testing_nesterov: Meta-testing Nesterov momentum.
        :param testing_batch_size: Meta-testing batch size.
        :param testing_milestones: Meta-testing scheduler milestones.
        :param testing_gamma: Meta-testing scheduler decay rate.
        :param device: Device used for Pytorch related components {"cpu", "cuda"}.
        :param random_state: Random state for reproducibility [N].
        :param verbose: Display console output at different levels {0, 1}.
        """

        # Setting the reproducibility seed in PyTorch.
        torch.cuda.manual_seed(random_state)
        torch.manual_seed(random_state)
        numpy.random.seed(random_state)
        random.seed(random_state)

        # Defining the representation and optimization.
        self.representation = representation
        self.optimization = optimization

        # Base network object and its corresponding parameters.
        self.base_network = base_network
        self.base_network_parameters = base_network_parameters

        # Meta-testing hyper-parameters.
        self.testing_gradient_steps = testing_gradient_steps
        self.testing_learning_rate = testing_learning_rate
        self.testing_weight_decay = testing_weight_decay
        self.testing_momentum = testing_momentum
        self.testing_nesterov = testing_nesterov
        self.testing_batch_size = testing_batch_size
        self.testing_milestones = testing_milestones
        self.testing_gamma = testing_gamma

        # Administrative hyper-parameters.
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

        # The meta-learned loss function object.
        self.meta_loss_function = None

    def meta_train(self, training_tasks, validation_tasks):

        """
        Executes the meta-training phase of the loss function learning algorithm. If a
        non population-based optimizer is given then initialization of the loss function
        occurs in this functions, otherwise if it is a population-based algorithm the
        optimizer handles initialization.

        :param training_tasks: List of training tasks, where a task is a PyTorch Dataset objects.
        :param validation_tasks: List of validation tasks, where a task is a PyTorch Dataset objects.
        :return Meta-learned loss function and a dictionary of meta-learning meta-data.
        """

        # Placing a singular task in a list to simulate multi-task meta-learning.
        if not isinstance(training_tasks, list):
            training_tasks = [training_tasks]
            validation_tasks = [validation_tasks]

        if self.representation == "gp" or self.optimization.is_population_based:
            self.meta_loss_function = None

        elif self.representation == "nn" and not self.optimization.is_population_based:
            self.meta_loss_function = nn.NeuralNetwork(**self.base_network_parameters).to(self.device)

        elif self.representation == "tp" and not self.optimization.is_population_based:
            self.meta_loss_function = tp.TaylorPolynomials(**self.base_network_parameters).to(self.device)

        if self.optimization.is_population_based:  # If optimizer is population-based.
            loss_function, history = self.optimization.train(
                self.representation, training_tasks, validation_tasks
            )

        else:  # If optimizer is not population-based.
            loss_function, history = self.optimization.train(
                self.meta_loss_function, training_tasks, validation_tasks
            )

        # Assigning the best loss function and returning it + the learning history.
        self.meta_loss_function = loss_function
        return loss_function, history

    def meta_test(self, tasks, performance_metric):

        """
        The meta-testing phase of the EvoMAL algorithm which is used to train the model at
        meta-testing time, using a vanilla training loop (i.e. no task loss and no high order
        derivative computation). Returns the performance metric and loss history over the
        tasks given.

        :param tasks: List of tasks, where a task is a PyTorch Dataset objects.
        :param performance_metric: Performance metric to use for evaluation.
        :return: Learning and inference history with the corresponding models.
        """

        # Placing a singular task in a list to simulate multi-task meta-learning.
        if not isinstance(tasks, list):
            tasks = [tasks]

        # List for storing all the trained models.
        models = []
        learning = []

        # For each task in the training set.
        for i in range(len(tasks)):

            # Resetting the random seed to ensure initialization is the same.
            torch.cuda.manual_seed(self.random_state)
            torch.manual_seed(self.random_state)

            # Creating a base network using the given base network parameters.
            network = self.base_network(**self.base_network_parameters).to(self.device)
            models.append(network)

            # Creating the gradient-based optimizer.
            optimizer = torch.optim.SGD(
                network.parameters(),
                lr=self.testing_learning_rate,
                momentum=self.testing_momentum,
                nesterov=self.testing_nesterov,
                weight_decay=self.testing_weight_decay
            )

            # Creating the learning rate scheduler.
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.testing_milestones,
                gamma=self.testing_gamma
            )

            # Executing the training session.
            learning.append(bp.backpropagation(
                model=network, dataset=tasks[i],
                loss_function=self.meta_loss_function,
                gradient_steps=self.testing_gradient_steps,
                batch_size=self.testing_batch_size,
                optimizer=optimizer, scheduler=scheduler,
                performance_metric=performance_metric,
                verbose=self.verbose, device=self.device,
                terminate_divergence=False
            ))

        return models, numpy.mean(learning, axis=0).tolist()

    def evaluate(self, models, tasks, performance_metric):

        """
        Evaluate a set of models on a set of tasks using a given performance
        metic. Returns a list of each tasks performance.

        :param models: Trained PyTorch models.
        :param tasks: Data to use for evaluating.
        :param performance_metric: Performance metric to use for evaluation.
        :return: List of each tasks performance.
        """

        # Placing a singular task in a list to simulate multi-task meta-learning.
        if not isinstance(tasks, list):
            tasks = [tasks]

        # List for storing all the trained models.
        results = []

        # For each task in the training set.
        for network, task in zip(models, tasks):

            # Performing inference on the network and saving the results.
            results.append(inference.evaluate(
                model=network,
                task=task,
                device=self.device,
                performance_metric=performance_metric
            ))

        return results

    def export_loss(self, file_path, file_name):

        """
        Exports the meta learned loss function to a .pth file. Function assumes that
        the meta-train function has been called already.

        :param file_path: Desired file path to export loss function to.
        :param file_name: Desired file name of exported loss function.
        """

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        torch.save(self.meta_loss_function, file_path + file_name + ".pth")

    def import_loss(self, file_path, file_name):

        """
        Imports a meta-learned loss function from a .pth file. Function assigns the
        learned loss function as the "meta loss function" enabling the meta-testing
        function of this object to be called again.

        :param file_path: Path to the imported loss function.
        :param file_name: Name of the imported loss function.
        """

        self.meta_loss_function = torch.load(file_path + file_name + ".pth")
        self.meta_loss_function.eval()
