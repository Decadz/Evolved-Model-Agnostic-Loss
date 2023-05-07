import torch


class LossFunctionFilters:

    def __init__(self, filter_gradient_steps, filter_sample_size, base_network,
                 base_network_parameters, learning_rate, momentum, nesterov,
                 performance_metric, device, **kwargs):

        """
        Utility class for the loss function rejection protocol and gradient
        equivalence filter.

        :param filter_gradient_steps: Gradient steps for optimizing the predictions.
        :param filter_sample_size: Number of predictions to sample.
        :param base_network: PyTorch base network object.
        :param base_network_parameters: List of parameters for the base network.
        :param learning_rate: Learning rate for optimizing the predictions.
        :param momentum: Momentum value for the optimizer.
        :param nesterov: Nesterov momentum for the optimizer.
        :param performance_metric: Performance metric to use for evaluation.
        :param device: Device used for Pytorch related components {"cpu", "cuda"}.
        """

        # Filter hyper-parameters and settings.
        self.filter_gradient_steps = filter_gradient_steps
        self.filter_sample_size = filter_sample_size
        self.base_network = base_network
        self.base_network_parameters = base_network_parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.performance_metric = performance_metric
        self.device = device

    def compute_predictions(self, tasks):
        pred_labels, true_labels = [], []

        for task in tasks:  # Iterating over all tasks in the meta-training set.

            # Creating a base network using the given base network parameters.
            network = self.base_network(**self.base_network_parameters).to(self.device)

            # Creating a PyTorch DataLoader to generate samples/batches for each tasks.
            generator = torch.utils.data.DataLoader(task, batch_size=self.filter_sample_size, shuffle=False)

            # Generating labels for a single batch then exiting.
            for instances, labels in generator:
                pred_labels.append(network(instances.to(self.device)))
                true_labels.append(labels.to(self.device))
                break

        return pred_labels, true_labels

    def rejection_protocol(self, loss_function, pred_labels, true_labels, task_type):

        difference_performance = []

        for pred, true in zip(pred_labels, true_labels):

            # Cloning the original vector for comparison later.
            baseline = pred.detach().clone().requires_grad_(False)
            predictions = pred.detach().clone().requires_grad_(True)

            # Creating an optimizer for optimizing the predictions.
            optimizer = torch.optim.Adam([predictions], lr=self.learning_rate)

            for step in range(self.filter_gradient_steps):

                # Clearing the gradient cache.
                optimizer.zero_grad()

                # Computing the loss value
                loss = loss_function(predictions, true)

                # Performing the backward pass and gradient step/update.
                loss.backward()
                optimizer.step()

            # Computing the difference between baseline and optimized.
            baseline = self.performance_metric(baseline, true)
            optimized = self.performance_metric(predictions, true)

            # Recording the difference between the baseline and optimized.
            difference_performance.append(baseline - optimized)

        reject = False  # Flag for determining whether the loss function should be rejected.

        if task_type == "classification":
            # Computing the average correlation across the samples.
            correlation = torch.mean(torch.stack(difference_performance))
            reject = True if correlation <= 0 else False
        elif task_type == "regression":
            correlation = torch.mean(torch.stack(difference_performance))
            reject = True if correlation <= 0 else False

        return reject
