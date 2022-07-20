# Evolved Model-Agnostic Loss

This repository contains code for reproducing the experiments in the paper 
"[*Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning*]()" by 
Christian Raymond, Qi Chen, Bing Xue, and Mengjie Zhang.

![banner-image](https://user-images.githubusercontent.com/23614094/179725386-b3189ce7-d81b-48d6-886d-e97745c788a4.png)


## Contents

A PyTorch + Higher implementation of the newly proposed *Evolved Model-Agnostic Loss* (EvoMAL) algorithm. In addition, there is code for reproducing loss function learning algorithms from the following papers:

* Meta-Learning via Learned Loss Supervised (Bechtle et al., 2021)
* Optimizing Loss Functions Through Multivariate Taylor Polynomial Parameterization (Gonzalez et al., 2021)
* Genetic Programming for Loss Function Learning (Liu et al., 2021), (Li et al., 2021)

## Installation

1. Clone this repository to your local machine:
```bash
git clone https://github.com/Decadz/Evolved-Model-Agnostic-Loss.git
cd Evolved-Model-Agnostic-Loss
```

2. Install the necessary libraries and dependencies:
```bash
pip install requirements.txt
```

## Usage

To meta-learn loss functions run the following command via the terminal:
```
python experiments/run_learning.py --method method_name --dataset dataset_name --model model_name --seeds [seeds] --device device
```

To transfer a previously meta-learned loss functions run the following command via the terminal:
```
python experiments/run_transfer.py --method method_name --source_dataset dataset_name --source_model model_name --source_seed seed --target_dataset dataset_name --target_model model_name --target_seeds [seeds] --device device
```

To generate loss landscapes visualizations run the following command via the terminal:
```
python experiments/run_landscapes.py --method method_name --dataset dataset_name --model model_name --distance distance --steps steps --seeds [seeds] --device device
```

### Available Arguments:

- **Method** = {baseline, ml3, taylorglo, gplfl, evomal}
- **Dataset** = {mnist, cifar10, cifar100, svhn}
- **Model** = {lenet5, alexnet, vgg, allcnnc, resnet, preresnet, wideresnet, squeezenet, pyramidnet}
  
### Additional Notes: 

* In addition to the compulsory arguments all hyper-parameters (found in the `configurations`) as well as the *device* can be given as optional arguments which will override the default values.
* The `run_landscapes.py` script generates loss landscapes which can be either visualized directly in PyVista or exported to a `.vtk` file which can be reopened in the open sources software ParaView.
* The meta-learned loss function and/or trained model used as a source for `run_transfer.py` or `run_landscapes.py` should be placed in the respective file in `experiments/results` in order for code to work as intended.

### Code Reproducibility: 

The code have not been comprehensively checked and re-run since refactoring. If you're having any issues, find
a problem/bug or cannot reproduce similar results as the paper please [open an issue](https://github.com/Decadz/Evolved-Model-Agnostic-Loss/issues)
or email me.

## Reference

If you use our library please consider citing our paper with the following Bibtex entry:

```
@inproceedings{raymond2022learning,
  title={Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning},
  author={Raymond, Christian and Chen, Qi and Xue, Bing and Zhang, Mengjie},
  booktitle={},
  pages={},
  year={2022},
  organization={}
}
```
