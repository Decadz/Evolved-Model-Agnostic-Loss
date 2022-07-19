from source.representation.utilities.gp_expression import operations
import torch


class GeneticProgrammingTree(torch.nn.Module):

    def __init__(self, expression_tree, adjacency_list, output_dim, parameterize, device,
                 reduction="mean", output_activation=torch.nn.Identity(), **kwargs):

        """
        Parameterized Genetic Programming expression tree which can be further optimized
        via the parameterized edges, i.e. the weights, between the nodes.

        :param expression_tree: DEAP expression tree.
        :param adjacency_list: Adjacency list (dictionary).
        :param output_dim: Dimensions of output from base network.
        :param parameterize Whether the expression should be parameterized for training.
        :param reduction: Reduction operator for aggregating results.
        :param output_activation: Loss function output activation.
        :param device: Device used for Pytorch related components {"cpu", "cuda"}.
        """

        super(GeneticProgrammingTree, self).__init__()

        self.expression_tree = expression_tree
        self.adjacency_list = adjacency_list
        self.output_dim = output_dim
        self.parameterize = parameterize
        self.reduction = reduction
        self.output_activation = output_activation
        self.device = device

        # Parameterizing the expression tree.
        for i in range(len(expression_tree)):
            if self.parameterize:
                param = torch.nn.Parameter(torch.randn(1, requires_grad=True).to(self.device) * 0.001 + 1)
            else:
                param = torch.nn.Parameter(torch.ones(1, requires_grad=False).to(self.device))

            # Registering the parameter with the torch module.
            self.register_parameter(name=str(i), param=param)

    def forward(self, y_pred, y_target):

        """
        Executing a forward pass (inference) on the meta-loss network. The forward pass makes
        use of a dictionary to calculate the intermediate values of the DAG.

        :param y_pred: PyTorch tensor of predictions made by the base-network (i.e. f(x)).
        :param y_target: PyTorch tensor of the true labels (i.e. y).
        :return: Average loss across the batch using the meta-loss network.
        """

        # Regression or binary classification problem.
        if self.output_dim == 1:
            loss = self._compute_loss(y_pred, y_target)

        else:  # Multi-class classification problem.
            y_target = torch.nn.functional.one_hot(y_target, num_classes=self.output_dim)
            loss = self._compute_loss(y_pred, y_target).sum(axis=1)

        # Applying the reduction operation to the loss vector.
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def _compute_loss(self, y_pred, y_target):

        """
        Executing a forward pass (inference) on the meta-loss network. The forward pass makes
        use of a dictionary to calculate the intermediate values of the DAG.

        :param y_pred: PyTorch tensor of predictions made by the base-network (i.e. f(x)).
        :param y_target: PyTorch tensor of the true labels (i.e. y).
        :return: Average loss across the batch using the meta-loss network.
        """

        results = {}

        # Iterating over the adjacency list in reverse order (from inputs to output).
        for key, values in reversed(list(self.adjacency_list.items())):

            # Extracting the parent node from the expression tree.
            parent_node = self.expression_tree[key]

            if parent_node.name == "ARG0":  # Terminal node "y" set value to the targets.
                results[key] = y_target

            elif parent_node.name == "ARG1":  # Terminal node "f(x)" set value to predictions.
                results[key] = y_pred

            elif parent_node.name == "1":  # Terminal node "+1" set value to a constant of 1.
                results[key] = torch.tensor([1], requires_grad=False).to(self.device)

            elif parent_node.name == "-1":  # Terminal node "-1" set value to a constant of -1.
                results[key] = torch.tensor([-1], requires_grad=False).to(self.device)

            # Evaluating each of the children nodes.
            for value in values:

                # Extracting the current child node from the expression tree.
                child_node = self.expression_tree[value]

                # If the function node has two arguments (i.e binary operator).
                if child_node.arity == 2:

                    if value not in results:  # If this is the first arguments evaluate branch.
                        results[value] = torch.mul(results[key], self._parameters[str(key)])

                    else:  # If this is the second argument evaluate the whole subtree.
                        results[value] = operations[child_node.name](
                            torch.mul(results[key], self._parameters[str(key)]),
                            results[value]
                        )

                # If the function node has one argument (i.e unary operator).
                elif child_node.arity == 1:
                    results[value] = operations[child_node.name](
                        torch.mul(results[key], self._parameters[str(key)])
                    )

        # Applying the output activation and returning the loss.
        return self.output_activation(results[0])

    def __str__(self):
        string, node_stack, param_stack = "", [], []

        # Iterating over all the nodes and weights.
        for i in range(len(self.expression_tree)):

            # Adding the current node and weight to their stacks.
            node_stack.append((self.expression_tree[i], []))
            param_stack.append(str(self._parameters[str(i)].item()))

            while len(node_stack[-1][1]) == node_stack[-1][0].arity:
                prim, args = node_stack.pop()

                if self.parameterize:
                    string = param_stack.pop() + "*" + prim.format(*args)
                else:
                    string = prim.format(*args)

                if len(node_stack) > 0:
                    node_stack[-1][1].append(string)
                else:
                    break

        # Adding the output activation to the string.
        if isinstance(self.output_activation, torch.nn.Softplus):
            string = "softplus(" + string + ")"

        return string
