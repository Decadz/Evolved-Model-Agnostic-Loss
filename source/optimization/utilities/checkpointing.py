from source.optimization.inference import evaluate


def model_checkpointing(meta_network, base_models, base_optimizers, training_task, validation_tasks,
                        base_gradient_steps, performance_metric, device):

    """
    Evaluating the meta-loss function by performing partial training sessions
    and then evaluating the validation error.

    :param meta_network: PyTorch meta-loss network.
    :param base_models: List of base networks for each task.
    :param base_optimizers: List of optimizers for each task.
    :param training_task: List of PyTorch training DataLoaders.
    :param validation_tasks: List of PyTorch validation Datasets.
    :param base_gradient_steps: Number of base gradient steps for validation.
    :param performance_metric: Performance metric to use for evaluation.
    :param device: Device used for Pytorch related components {"cpu", "cuda"}.
    :return: The average validation performance.
    """

    validation_performance = []

    # For each training task in the task distribution.
    for task in range(len(base_models)):

        # Performing a partial training session.
        for step in range(base_gradient_steps):

            # Clearing the gradient cache.
            base_optimizers[task].zero_grad()

            # Sampling a mini batch from the task.
            X, y = next(iter(training_task[task]))

            # Sending data to the correct device.
            X = X.to(device)
            y = y.to(device)

            # Performing inference and computing the loss.
            y_pred = base_models[task](X)
            loss = meta_network(y_pred, y)

            # Performing the backward pass and gradient step/update.
            loss.backward()
            base_optimizers[task].step()

        # Evaluating the base model on the validation set.
        validation_performance.append(evaluate(
            base_models[task], validation_tasks[task], device,
            performance_metric=performance_metric
        ))

    return sum(validation_performance) / len(validation_performance)
