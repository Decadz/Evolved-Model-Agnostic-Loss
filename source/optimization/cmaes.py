from source.representation.tp import TaylorPolynomials
from source.optimization.bp import backpropagation
from source.optimization.inference import evaluate

from deap import creator
from deap import base
from deap import cma

from tqdm import tqdm

import numpy as np
import torch


# Defining the DEAP optimization object.
creator.create("MinimizeFit", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.MinimizeFit)


class CovarianceMatrixAdaptation:

    def __init__(self, base_network, base_network_parameters, population_size, num_generations,
                 filter_significant_digits, filter_rejection_threshold, filter_gradient_steps,
                 filter_sample_size, base_gradient_steps, base_learning_rate, base_weight_decay,
                 base_momentum, base_nesterov, base_batch_size, performance_metric, random_state,
                 verbose, device, **kwargs):

        """
        Implementation of a covariance matrix adaptation evolutionary strategy. This
        module specifically, is an evolutionary strategy for the cubic taylor-polynomial
        representation.

        :param base_network: PyTorch base network object.
        :param base_network_parameters: List of parameters for the base network.
        :param local_search: Local search mechanism (optimization object).
        :param population_size: Population size [N+].
        :param num_generations: Number of generations [N+].
        :param filter_significant_digits: Number of significant digits in gradient equivalence.
        :param filter_rejection_threshold: Rejection threshold in rejection protocol.
        :param base_gradient_steps: Number of gradient for partial training session.
        :param base_learning_rate: Base learning rate for the base network/s.
        :param base_weight_decay: Base weight decay for the base network/s.
        :param base_momentum: Base momentum for the base network/s.
        :param base_nesterov: Base nesterov momentum for the base network/s.
        :param base_batch_size: Batch size for each gradient step.
        :param performance_metric: Performance metric to use for evaluation.
        :param random_state: Random state for reproducibility [N].
        :param verbose: Display console output at different levels {0, 1}.
        :param device: Device used for Pytorch related components {"cpu", "cuda"}.
        """

        # Flag declaring whether this is a population based optimizers.
        self.is_population_based = True

        # Defining the meta-learning objects and settings.
        self.base_network = base_network
        self.base_network_parameters = base_network_parameters

        # Explicit GP hyper-parameters.
        self.population_size = population_size
        self.num_generations = num_generations

        # Evolutionary filter hyper-parameters.
        self.significant_digits = filter_significant_digits
        self.rejection_threshold = filter_rejection_threshold

        # Fitness evaluation hyper-parameters.
        self.base_gradient_steps = base_gradient_steps
        self.base_learning_rate = base_learning_rate
        self.base_weight_decay = base_weight_decay
        self.base_momentum = base_momentum
        self.base_nesterov = base_nesterov
        self.base_batch_size = base_batch_size

        # Recording the administrative hyper-parameters.
        self.performance_metric = performance_metric
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

    def train(self, representation, training_tasks, validation_tasks):

        if representation != "tp":
            raise ValueError("Only Taylor polynomial representation is supported.")

        # Defining the CMA-ES initialization and update rules.
        strategy = cma.Strategy(centroid=[0] * 8, sigma=1.2, lambda_=self.population_size)

        # Variables for keeping track of the learning process.
        best_loss_network = None
        best_loss_fitness = None
        training_history = []

        for gen in tqdm(range(self.num_generations), desc="Evolution Progression", position=0,
                        disable=False if self.verbose >= 1 else True, leave=False):

            # Generate a new population of candidate solutions.
            population = strategy.generate(ind_init=creator.Individual)

            if gen != 0:  # Updating the population.
                strategy.update(population)

            # Training each expression tree in the population.
            for solution in tqdm(population, desc="Generation Progression", position=1,
                                 disable=False if self.verbose >= 2 else True, leave=False):

                # Generating the meta-network for training.
                meta_loss_function = TaylorPolynomials(self.base_network_parameters["output_dim"], self.device)
                for index, weight in enumerate(solution):
                    weight_tensor = torch.nn.Parameter(torch.tensor(weight, requires_grad=False))
                    meta_loss_function.register_parameter(name=str(index), param=weight_tensor)

                # Evaluating the fitness of the meta loss function.
                fitness = self._fitness_evaluation(meta_loss_function, training_tasks, validation_tasks)
                solution.fitness.values = [fitness]

                # Updating the best expression/loss found by EvoMAL.
                if best_loss_network is None or np.isnan(best_loss_fitness) or \
                        solution.fitness.values[0] < best_loss_fitness:
                    best_loss_network = meta_loss_function
                    best_loss_fitness = fitness

            # Recording the best fitness found thus far in the search.
            training_history.append(best_loss_fitness)

        return best_loss_network, {"meta-training": training_history}

    def _fitness_evaluation(self, loss_function, training_tasks, validation_tasks):

        fitness_scores = []

        for train, valid in zip(training_tasks, validation_tasks):

            # Resetting the random seed to ensure initialization is the same.
            torch.cuda.manual_seed(self.random_state)
            torch.manual_seed(self.random_state)

            # Creating a new base network.
            network = self.base_network(**self.base_network_parameters).to(self.device)

            # Creating the gradient-based optimizer.
            optimizer = torch.optim.SGD(
                network.parameters(),
                lr=self.base_learning_rate,
                momentum=self.base_momentum,
                nesterov=self.base_nesterov,
                weight_decay=self.base_weight_decay
            )

            # Creating the learning rate scheduler.
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0)

            # Executing a partial training session.
            backpropagation(
                model=network, dataset=train, loss_function=loss_function,
                gradient_steps=self.base_gradient_steps, batch_size=self.base_batch_size,
                optimizer=optimizer, scheduler=scheduler, verbose=0, device=self.device,
                performance_metric=self.performance_metric
            )

            # Obtaining the inference performance on the validation set.
            fitness_scores.append(evaluate(
                model=network, task=valid, device=self.device,
                performance_metric=self.performance_metric)
            )

        return sum(fitness_scores) / len(fitness_scores)
