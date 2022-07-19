from source.representation.utilities.phase_transformer import phase_transformer
from source.representation.utilities.gp_expression import pset
from source.optimization.utilities.filters import LossFunctionFilters
from source.optimization.bp import backpropagation
from source.optimization.inference import evaluate

from copy import deepcopy
from deap import creator
from deap import tools
from deap import gp

from functools import partial
from tqdm import tqdm

import random as rd
import numpy as np
import torch
import math


class EvolutionaryAlgorithm:

    def __init__(self, base_network, base_network_parameters, local_search, population_size,
                 num_generations, crossover_rate, mutation_rate, elitism_rate, tournament_size,
                 max_height, init_min_height, init_max_height, filter_significant_digits,
                 filter_rejection_threshold, filter_gradient_steps, filter_sample_size,
                 base_gradient_steps, base_learning_rate, base_weight_decay, base_momentum,
                 base_nesterov, base_batch_size, performance_metric, random_state, verbose,
                 device, **kwargs):

        """
        Implementation of a vanilla evolutionary algorithm. This module specifically,
        is an evolutionary algorithm for the genetic programming expression tree-based
        representation.

        :param base_network: PyTorch base network object.
        :param base_network_parameters: List of parameters for the base network.
        :param local_search: Local search mechanism (optimization object).
        :param population_size: Population size [N+].
        :param num_generations: Number of generations [N+].
        :param crossover_rate: Crossover rate [0, 1].
        :param mutation_rate: Mutation rate [0, 1].
        :param elitism_rate: Elitism rate [0, 1].
        :param tournament_size: Tournament selection size [N+].
        :param max_height: Max expression tree height [N+].
        :param init_min_height: Minimum mutation expression tree height [N+].
        :param init_max_height: Maximum mutation expression tree height [N+].
        :param filter_significant_digits: Number of significant digits in gradient equivalence.
        :param filter_rejection_threshold: Rejection threshold in rejection protocol.
        :param filter_gradient_steps: Gradient steps for optimizing the predictions.
        :param filter_sample_size: Number of predictions to sample.
        :param base_gradient_steps: Number of gradient for partial training session.
        :param base_learning_rate: Base learning rate for the base network/s.
        :param base_weight_decay: Base weight decay for the base network/s.
        :param base_momentum: Base momentum for the base network/s.
        :param base_nesterov: Base Nesterov momentum for the base network/s.
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
        self.local_search = local_search
        self.parameterize = False if local_search is None else True

        # Explicit GP hyper-parameters.
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size

        # Evolutionary filter hyper-parameters.
        self.significant_digits = filter_significant_digits
        self.rejection_threshold = filter_rejection_threshold

        # Implicit GP hyper-parameters.
        self.max_height = max_height
        self.init_min_height = init_min_height
        self.init_max_height = init_max_height

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

        # Generate a loss function filer object.
        self.loss_filters = LossFunctionFilters(
            filter_gradient_steps=filter_gradient_steps,
            filter_sample_size=filter_sample_size,
            base_network=base_network,
            base_network_parameters=base_network_parameters,
            learning_rate=base_learning_rate,
            momentum=base_momentum,
            nesterov=base_nesterov,
            performance_metric=performance_metric,
            device=device
        )

    def train(self, representation, training_tasks, validation_tasks):

        """
        Performs the meta-training phase using an evolutionary algorithm to optimize the given
        meta-loss function representation.

        :param representation: Representation use for the meta-loss function.
        :param training_tasks: List of training tasks, where a task is a PyTorch Dataset objects.
        :param validation_tasks: List of validation tasks, where a task is a PyTorch Dataset objects.
        :return: Final loss meta network and a dictionary containing the meta-training meta-data.
        """

        if representation != "gp":
            raise ValueError("Only GP representation is supported.")

        # Variables for keeping track of the learning process.
        best_loss_network = None
        best_loss_fitness = None
        training_history = []

        # Archive for keeping track of seen loss functions and their fitness.
        symbolic_equivalence_archive = {}
        gradient_equivalence_archive = {}

        # List for keeping track of loss function filter performance.
        symbolic_equivalence_history = [0] * self.num_generations
        rejection_protocol_history = [0] * self.num_generations
        gradient_equivalence_history = [0] * self.num_generations

        # Creating the initial population of loss functions.
        population = self._initialize_population()

        # Iterating over a predefined number of generations.
        for gen in tqdm(range(self.num_generations), desc="Evolution Progression", position=0,
                        disable=False if self.verbose >= 1 else True, leave=False):

            if gen != 0:  # Evolve a new population of solutions if it's not the first generation.
                population = self._evolve(population, self.population_size)

            # Training each expression tree in the population.
            for expression in tqdm(population, desc="Generation Progression", position=1,
                                   disable=False if self.verbose >= 2 else True, leave=False):

                # Evaluating the symbolic equivalence filter.
                if symbolic_equivalence_archive.get(str(expression)):
                    expression.fitness.values = symbolic_equivalence_archive[str(expression)]
                    symbolic_equivalence_history[gen] += 1
                    continue

                # Converting the expression tree into a trainable PyTorch network.
                meta_network = phase_transformer(
                    expression_tree=expression,
                    output_dim=self.base_network_parameters["output_dim"],
                    parameterize=self.parameterize,
                    device=self.device
                )

                # Performing the meta-training backpropagation stage.
                if self.local_search is not None:
                    self.local_search.train(meta_network, training_tasks, None)

                # Generating a sample of predicted and true labels for the following filter.
                pred_labels, true_labels = self.loss_filters.compute_predictions(training_tasks)

                # Computing the rejection protocol correlation and final prediction vector.
                correlation, predictions = self.loss_filters.rejection_protocol(meta_network, pred_labels, true_labels)

                # Evaluating the rejection protocol filter.
                if correlation <= self.rejection_threshold:
                    rejection_protocol_history[gen] += 1
                    expression.fitness.values = [float("inf")]
                    continue

                # Computing the mean gradient norm of the prediction vector.
                gradient_norm = self.loss_filters.gradient_equivalence(predictions)

                # Comparing against all previous solutions
                for key, value in gradient_equivalence_archive.items():

                    # Comparing the current norm values to all the others.
                    compare = [math.isclose(a, b, abs_tol=10 ** -2)for a, b in zip(gradient_norm, value)]

                    # If all values are equivalent up to 2 decimal points.
                    if not compare.__contains__(False):
                        expression.fitness.values = symbolic_equivalence_archive[key]
                        gradient_equivalence_history[gen] += 1
                        break

                # If the fitness function has not yet been set, evaluate it.
                if not expression.fitness.values:

                    # Evaluating the meta learned loss functions fitness.
                    fitness = self._fitness_evaluation(meta_network, training_tasks, validation_tasks)
                    symbolic_equivalence_archive[str(expression)] = [fitness]
                    gradient_equivalence_archive[str(expression)] = gradient_norm
                    expression.fitness.values = [fitness]

                # Updating the best expression/loss found by EvoMAL.
                if best_loss_network is None or np.isnan(best_loss_fitness) or \
                        expression.fitness.values[0] < best_loss_fitness:

                    best_loss_network = meta_network
                    best_loss_fitness = fitness

            # Recording the best fitness found thus far in the search.
            training_history.append(best_loss_fitness)

        return best_loss_network, {
            "meta-training": training_history,
            "symbolic_equivalence_history": symbolic_equivalence_history,
            "rejection_protocol_history": rejection_protocol_history,
            "gradient_equivalence_history": gradient_equivalence_history
        }

    def _evolve(self, population, num_offspring):

        """
        Evolves the population using a standard Evolutionary Algorithm (EA) approach.
        Returns a new population of expression trees, which all have had constraints
        enforced (contain both y and f(x) terminals). This uses the common varOr
        variant (as opposed to varAnd), which perform either crossover or mutation.

        :param population: Population of solutions.
        :param num_offspring: Number of offspring to generate.
        """

        # List containing the offspring (new) population.
        offspring = []

        # Using a double tournament selection to select expression trees from the population.
        selected = tools.selDoubleTournament(
            individuals=population,
            k=num_offspring,
            fitness_size=self.tournament_size,
            parsimony_size=1,
            fitness_first=False
        )

        # Generating new individuals for the offspring population.
        for i in range(len(selected)):
            rand = rd.random()

            # Selecting the base expression tree to evolve.
            ind = deepcopy(selected[i])

            # Apply the crossover genetic operator.
            if rand < self.crossover_rate:
                ind, _ = gp.cxOnePoint(ind, deepcopy(rd.choice(selected)))

            # Apply the mutation genetic operator.
            elif rand < self.crossover_rate + self.mutation_rate:
                ind = gp.mutUniform(ind, partial(gp.genHalfAndHalf, pset=pset, min_=0, max_=2), pset)[0]

            # If expression is to large reject it and use reproduction.
            if ind.height > self.max_height:
                offspring.append(deepcopy(selected[i]))

            # If expression is valid add it to the population.
            else:
                offspring.append(ind)
                del ind.fitness.values

        # Assigning the current population to the new offspring.
        population = offspring

        # Applying all the expression tree preprocessing steps.
        self._apply_constraints(population)

        return population

    def _initialize_population(self):

        """
        Generating a predefined number [population_size] of expression trees with the
        popular ramped half and half initialization method. Returns a list of DEAP
        expression tree objects, with a min size (depth) of [init_min_height] and max
        size of [init_max_height].
        """

        # Generating a list of expression trees using ramped half and half.
        population = [creator.Expression(gp.genHalfAndHalf(pset, self.init_min_height,
                        self.init_max_height)) for _ in range(self.population_size)]

        # Applying all the expression tree preprocessing steps.
        self._apply_constraints(population)

        # Returning the newly generated population.
        return population

    def _apply_constraints(self, population):

        """
        Checks all expression tree in the population to ensure the constraint
        requiring both y and f(x) nodes is meet. If it is not, a random terminal
        node is mutated to contain a random binary operator which contains both
        y and f(x) as arguments.
        """

        # Checking each expression tree to ensure constraints are meet.
        for expression in population:

            # Flags to check for if the expression tree contains the required nodes.
            contains_y, contains_fx = False, False

            # Lists of indexes where terminal nodes appear.
            terminal_indexes = []

            # Iterating over all the nodes in the expression
            for index, node in enumerate(expression):
                if isinstance(node, gp.Terminal):

                    # Recording the index of the current terminal node.
                    terminal_indexes.append(index)

                    # Checking if the current node is y or f(x).
                    if node.value == "ARG0":
                        contains_y = True

                    elif node.value == "ARG1":
                        contains_fx = True

                # If the constraints are meet stop checking over all the nodes.
                if contains_y and contains_fx:
                    break

            # If the constraints are meet go to the next expression tree.
            if contains_y and contains_fx:
                continue

            # Getting the required terminal nodes, and a random binary operator.
            binary_op = pset.primitives[object][rd.randrange(4)]
            args = [pset.terminals[object][0], pset.terminals[object][1]]
            rd.shuffle(args)

            # Selecting a random index and replacing it with the new subtree.
            index = terminal_indexes[rd.randrange(len(terminal_indexes))]
            expression[index:index] = [binary_op, args[0], args[1]]
            del expression[index + 3]

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

        return sum(fitness_scores)/len(fitness_scores)
