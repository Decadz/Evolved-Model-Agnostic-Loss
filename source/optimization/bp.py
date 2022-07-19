import torch
import numpy

from tqdm import tqdm


def backpropagation(model, dataset, loss_function, gradient_steps, batch_size,
                    optimizer, scheduler, performance_metric, verbose, device,
                    terminate_divergence=True):

    """
    A vanilla training loop which uses stochastic gradient descent to learn the
    parameters of the base network, using the given pytorch loss function.

    :param model: Base network used for the given task.
    :param dataset: PyTorch Dataset containing the training data.
    :param loss_function: Loss function to minimize.
    :param gradient_steps: Number of maximum gradient steps.
    :param batch_size: Backpropagation batch size.
    :param optimizer: Backpropagation gradient optimizer.
    :param scheduler: PyTorch learning rate scheduler.
    :param performance_metric: Performance metric to use for evaluation.
    :param verbose: Display console output at different levels {0, 1}.
    :param device: Device used for Pytorch related computation.
    :param terminate_divergence: Boolean for if divergent training is terminated.
    :return: List containing the meta-training history.
    """

    # Creating a PyTorch DataLoader to generate samples/batches for each tasks.
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    training_history = []

    # Looping until the maximum number of gradient steps is reached.
    for step in tqdm(range(gradient_steps), desc="Backpropagation Progression", position=0,
                     disable=False if verbose >= 1 else True, leave=False):

        # If the predetermined number of gradient steps has been reached.
        if step >= gradient_steps:
            break

        # Clearing the gradient cache.
        optimizer.zero_grad()

        # Sampling a mini batch from the task.
        X, y = next(iter(generator))

        # Sending data to the correct device.
        X = X.to(device)
        y = y.to(device)

        # Performing inference and computing the loss.
        y_pred = model(X)
        loss = loss_function(y_pred, y)

        # Terminating training if an invalid loss is achieved.
        if terminate_divergence:
            if torch.isnan(loss) or torch.isinf(loss):
                break

        # Performing the backward pass and gradient step/update.
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Recording the training performance.
        training_history.append(performance_metric(y_pred, y).item())

    return training_history
