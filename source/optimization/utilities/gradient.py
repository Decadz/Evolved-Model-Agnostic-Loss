
def apply_gradients(model, gradients):

    """
    Utility function which populates the parameters of a given torch module
    (i.e. model) with the corresponding gradient(s).

    :param model: Torch module (i.e. model).
    :param gradients: Gradients for the model.
    """

    offset = 0
    for p in model.parameters():
        p.grad = gradients[offset:offset + p.nelement()].view(p.size())
        offset += p.nelement()
