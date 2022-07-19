import torch


class MultiErrorRate:

    def __call__(self, pred, true):
        correct, incorrect = 0, 0

        for fx, y in zip(pred, true):
            if y == torch.argmax(fx):
                correct += 1
            else:
                incorrect += 1

        return torch.tensor(1 - (correct / (correct + incorrect)))


class BinaryErrorRate:

    def __call__(self, pred, true):
        correct, incorrect = 0, 0

        for fx, y in zip(pred, true):
            if y == 0 and fx < 0.5:
                correct += 1
            elif y == 1 and fx >= 0.5:
                correct += 1
            else:
                incorrect += 1

        return torch.tensor(1 - (correct / (correct + incorrect)))
