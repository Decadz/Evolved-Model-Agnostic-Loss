import torch
import os


def export_results(results, res_directory, file_name):

    # Exporting the results to a json file.
    if not os.path.exists(res_directory):
        os.makedirs(res_directory)

    _export_file(results, res_directory + file_name + ".json")


def export_loss(loss_function, res_directory, file_name):

    # Exporting the results to a json file.
    if not os.path.exists(res_directory):
        os.makedirs(res_directory)

    if loss_function is not None:  # Exporting the meta-learned loss function to a .pth file.
        if not os.path.exists(res_directory + "loss_functions/"):
            os.makedirs(res_directory + "loss_functions/")
        torch.save(loss_function, res_directory + "loss_functions/" + file_name + ".pth")


def export_model(model, res_directory, file_name):

    # Exporting the results to a json file.
    if not os.path.exists(res_directory):
        os.makedirs(res_directory)

    if model is not None:  # Exporting the trained model to a .pth file.
        if not os.path.exists(res_directory + "models/"):
            os.makedirs(res_directory + "models/")
        torch.save(model, res_directory + "models/" + file_name + ".pth")


def _export_file(results, file_name):

    """
    Takes a dictionary containing the results and experimental configurations
    and exports it to a .json file in the desired directory. If the directory
    does not exist it is created.

    :param results: Dictionary containing the experimental results.
    :param file_name: File name for the output results file.
    """

    # Removing administrative information.
    del results["experiment_configuration"]["output_path"]
    del results["experiment_configuration"]["verbose"]

    def _format_dictionary(dictionary, level=0, indent=4):
        string = ""
        if isinstance(dictionary, dict):
            string += "{" + "\n"
            comma = ""
            for key, value in dictionary.items():
                string += comma
                comma = ",\n"
                string += " " * indent * (level + 1)
                string += '"' + str(key) + '":' + " "
                string += _format_dictionary(value, level + 1)

            string += "\n" + " " * indent * level + "}"
        elif isinstance(dictionary, str):
            string += '"' + dictionary + '"'
        elif isinstance(dictionary, list):
            string += "[" + ",".join([_format_dictionary(e, level + 1) for e in dictionary]) + "]"
        elif isinstance(dictionary, tuple):
            string += "[" + ",".join(_format_dictionary(e, level + 1) for e in dictionary) + "]"
        elif isinstance(dictionary, bool):
            string += "true" if dictionary else "false"
        elif isinstance(dictionary, int):
            string += str(dictionary)
        elif isinstance(dictionary, float):
            string += "%.7g" % dictionary
        elif dictionary is None:
            string += "null"
        else:
            raise TypeError("Unknown type '%s' for json" % str(type(dictionary)))
        return string

    # Exporting all the results and information to a .json file.
    file = open(file_name, "w+")
    file.write(_format_dictionary(results))
    file.close()
