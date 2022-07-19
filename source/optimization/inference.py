import torch


def evaluate(model, task, device, performance_metric, batch_size=100):

    """
    Performs inference on the provided model, and computes the
    performance using the provided performance metric.

    :param model: Base network used for the given task.
    :param task: PyTorch Dataset used for evaluation.
    :param device: Device used for Pytorch related computation.
    :param performance_metric: Performance metric to use for evaluation.
    :param batch_size: Batch size used for inference.
    """

    # Creating a PyTorch DataLoader to generate samples/batches for each tasks.
    generator = torch.utils.data.DataLoader(task, batch_size=batch_size, shuffle=False)

    pred_labels, true_labels = [], []

    model.eval()  # Switching network to inference mode.
    with torch.no_grad():  # Disabling gradient calculations.

        # Iterating over the whole dataset in batches.
        for instances, labels in generator:
            pred_labels.append(model(instances.to(device)))
            true_labels.append(labels.to(device))

    # Converting the list to a PyTorch tensor.
    pred_labels = torch.cat(pred_labels)
    true_labels = torch.cat(true_labels)

    model.train()  # Switching network back to training mode.

    # Returning the performance of the trained model.
    return performance_metric(pred_labels, true_labels).item()
