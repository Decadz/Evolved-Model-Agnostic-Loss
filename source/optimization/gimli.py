import torch
import higher

from tqdm import tqdm


class GeneralizedInnerLoopMetaLearning:

    def __init__(self, base_network, base_network_parameters, task_loss, meta_gradient_steps,
                 meta_learning_rate, inner_gradient_steps, base_gradient_steps, base_learning_rate,
                 base_weight_decay, base_momentum, base_nesterov, base_batch_size, performance_metric,
                 random_state, verbose, device, **kwargs):

        """
        "Generalized Inner Loop Meta Learning" (GIMLI) by Grefenstette et al. or more specifically
        the optimization method presented in "Meta Learning via Learned Loss" by Bechtle et al.

        :param base_network: PyTorch base network object.
        :param base_network_parameters: List of parameters for the base network.
        :param task_loss: Task loss to use for the inner optimization.
        :param meta_gradient_steps: BP number of gradient steps for EvoMAL.
        :param meta_learning_rate: Meta-learning rate for the meta optimizer.
        :param inner_gradient_steps: Number of inner gradient steps before reset.
        :param base_gradient_steps: Number of base gradient steps for validation.
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
        self.is_population_based = False

        self.base_network = base_network
        self.base_network_parameters = base_network_parameters
        self.task_loss = task_loss
        self.meta_gradient_steps = meta_gradient_steps
        self.meta_learning_rate = meta_learning_rate
        self.inner_gradient_steps = inner_gradient_steps
        self.base_gradient_steps = base_gradient_steps
        self.base_learning_rate = base_learning_rate
        self.base_weight_decay = base_weight_decay
        self.base_momentum = base_momentum
        self.base_nesterov = base_nesterov
        self.base_batch_size = base_batch_size
        self.performance_metric = performance_metric

        # Recording the administrative hyper-parameters.
        self.random_state = random_state
        self.verbose = verbose
        self.device = device

    def train(self, meta_network, training_tasks, validation_tasks):

        """
        Performs the meta-training phase using generalized inner loop meta-learning to
        optimize the given meta-loss function representation.

        :param meta_network: Pytorch meta-loss network.
        :param training_tasks: List of training tasks, where a task is a PyTorch Dataset objects.
        :param validation_tasks: List of validation tasks, where a task is a PyTorch Dataset objects.
        :return: Final loss meta network and a dictionary containing the meta-training .
        """

        # List of base models and their respective optimizers.
        training_task_generators, validation_task_generators = [], []
        base_models, base_optimizers = [], []

        # Iterating over each task in the meta-training set.
        for i in range(len(training_tasks)):

            # Resetting the random seed to ensure initialization is the same.
            torch.cuda.manual_seed(self.random_state)
            torch.manual_seed(self.random_state)

            # Creating a new base network for each task in the training set.
            network = self.base_network(**self.base_network_parameters).to(self.device)
            base_models.append(network)

            # Creating a new optimizer for each base network.
            base_optimizers.append(torch.optim.SGD(
                base_models[i].parameters(), lr=self.base_learning_rate,
                momentum=self.base_momentum, nesterov=self.base_nesterov,
                weight_decay=self.base_weight_decay
            ))

            # Creating a PyTorch DataLoader to generate samples/batches for each tasks.
            training_task_generators.append(torch.utils.data.DataLoader(
                training_tasks[i], batch_size=self.base_batch_size, shuffle=True))

            validation_task_generators.append(torch.utils.data.DataLoader(
                validation_tasks[i], batch_size=self.base_batch_size, shuffle=True))

        # Performing the generalized inner loop meta-learning.
        training_history = self._inner_loop_learning(
            meta_network, base_models, base_optimizers,
            training_task_generators, validation_task_generators
        )

        return meta_network, {"meta-training": training_history}

    def _inner_loop_learning(self, meta_network, base_models, base_optimizers,
                             training_task_generators, validation_task_generators):

        # Defining the outer optimizer for the meta-loss network.
        meta_optimizer = torch.optim.Adam(meta_network.parameters(), lr=self.meta_learning_rate)

        # List for keeping track of the learning history.
        training_history = []

        # Performing the offline initialization phase to learn the learned loss functions parameters (phi).
        for step in tqdm(range(self.meta_gradient_steps), desc="GIMLI Progression", position=0,
                         disable=False if self.verbose >= 1 else True, leave=False):

            # Resetting the weights of all the base models.
            for i in range(len(training_task_generators)):
                base_models[i].reset()

            # Clearing the gradient cache.
            meta_optimizer.zero_grad()

            # List for keeping track of the learning history of each task.
            task_history = []

            # For each training task in the task distribution.
            for i in range(len(training_task_generators)):

                # Creating a differentiable optimizer and stateless models via PyTorch higher.
                with higher.innerloop_ctx(base_models[i], base_optimizers[i],
                                          copy_initial_weights=False) as (fmodel, diffopt):

                    # Taking a predetermined number of inner steps before meta update.
                    for inner_steps in range(self.inner_gradient_steps):

                        # Extracting a new training batch from the current task.
                        X_train, y_train = next(iter(training_task_generators[i]))
                        X_train, y_train = X_train.to(self.device), y_train.to(self.device)

                        # Computing the loss using the learned loss and updating the base weights.
                        yp_train = fmodel(X_train)  # Computing the base network predictions.
                        base_loss = meta_network(yp_train, y_train)  # Finding the loss wrt. learned-loss.
                        diffopt.step(base_loss)  # Update base network weights (theta).

                    # Extracting a validation batch from the current task.
                    X_train, y_train = next(iter(validation_task_generators[i]))
                    X_train, y_train = X_train.to(self.device), y_train.to(self.device)

                    # Computing predictions on the validation sets.
                    yp_train = fmodel(X_train)  # Predictions with new weights on the validation set.

                    # Computing the task loss and updating the meta weights.
                    task_loss = self.task_loss(yp_train, y_train)  # Finding the loss wrt. meta (task) loss.
                    task_loss.backward()  # Accumulates gradients wrt. to meta parameters.

                    # Storing the validation performance history.
                    task_history.append(self.performance_metric(yp_train, y_train).item())

                # Appending the average (mean) task loss to the tracker.
                training_history.append(sum(task_history) / len(training_task_generators))

            # Update meta-loss network weights (phi).
            meta_optimizer.step()

        return training_history
