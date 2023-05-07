from source.representation.gp import GeneticProgrammingTree


def phase_transformer(expression_tree, output_dim, parameterize, logits_to_prob, one_hot_encode, device):

    """
    Converts a DEAP expression tree and converts it into a PyTorch network.

    :param expression_tree: DEAP expression tree.
    :param output_dim: Dimensions of output from base network.
    :param parameterize Whether the expression should be parameterized for training.
    :param logits_to_prob: Apply transform to convert predicted output to probability.
    :param one_hot_encode: Apply transform to convert label to one-hot encoded label.
    :param device: Device used for Pytorch related components {"cpu", "cuda"}.
    :return: PyTorch meta-loss network.
    """

    # Creates and transposes an adjacency list using the given expression tree.
    adjacency_list = _create_adjacency_list(expression_tree)
    adjacency_list = _transpose_graph(adjacency_list)

    # Takes the adjacency list and creates a PyTorch network (DAG).
    return GeneticProgrammingTree(expression_tree, adjacency_list, output_dim, parameterize, device,
                                  logits_to_prob=logits_to_prob, one_hot_encode=one_hot_encode)


def _create_adjacency_list(expression_tree):

    """
    Converts a DEAP expression tree into an adjacency list, which is internally
    represented as a dictionary, where the key is the parent nodes index in the
    expression and the values are a list of the indexes of the children nodes.

    :param expression_tree: DEAP expression tree.
    :return: Adjacency list (dictionary).
    """

    adjacency_list = {}

    for i in range(len(expression_tree)):

        # If its a terminal don't add any new connections.
        if expression_tree[i].arity == 0:
            continue

        # Adding the unary operator and its child node to the adjacency list.
        elif expression_tree[i].arity == 1:
            arg = expression_tree.searchSubtree(i + 1)
            adjacency_list[i] = [arg.start]

        # Adding the binary operator and its children node to the adjacency list.
        elif expression_tree[i].arity == 2:
            arg1 = expression_tree.searchSubtree(i + 1)
            arg2 = expression_tree.searchSubtree(arg1.stop)
            adjacency_list[i] = [arg1.start, arg2.start]

    return adjacency_list


def _transpose_graph(adjacency_list):

    """
    Takes a (directed acyclic) graph represented as an adjacency list and transposes
    it. Returns a new adjacency list which represents the same graph, but with the
    directed edges reversed. An adjacency list is used as opposed to an adjacency
    matrix due to the more efficient asymptotic complexity of O(|V|+|E|) vs O(|V|^2).

    :param adjacency_list: Adjacency list (dictionary).
    :return: Transposed adjacency list (dictionary).
    """

    transposed_adjacency_list = {}

    # Iterating over each element in the adjacency list.
    for node, children in adjacency_list.items():

        # Iterating over each child node.
        for child in children:

            # Getting the list of children in the transposed graph.
            transposed_children = transposed_adjacency_list.get(child)

            if not transposed_children:
                # If list is empty initialize list with (parent) node.
                transposed_adjacency_list[child] = [node]
            else:
                # Else append (parent) node to pre-existing list.
                transposed_children.append(node)

    return transposed_adjacency_list
