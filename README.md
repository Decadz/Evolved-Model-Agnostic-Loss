# Evolved Model-Agnostic Loss

This repository contains code for reproducing the experiments in the paper 
"[*Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning*](https://arxiv.org/abs/2209.08907)" by 
Christian Raymond, Qi Chen, Bing Xue, and Mengjie Zhang.

![banner-image](https://user-images.githubusercontent.com/23614094/179725386-b3189ce7-d81b-48d6-886d-e97745c788a4.png)


## Contents

A [PyTorch](https://pytorch.org/) + [Higher](https://github.com/facebookresearch/higher) implementation of the newly proposed *Evolved Model-Agnostic Loss* (EvoMAL) algorithm. In addition, there is also code for reproducing loss function learning algorithms from the following papers:

* Meta-Learning via Learned Loss Supervised ([Bechtle et al., 2021](https://arxiv.org/abs/1906.05374))
* Optimizing Loss Functions Through Multivariate Taylor Polynomial Parameterization ([Gonzalez et al., 2021](https://arxiv.org/abs/2002.00059))
* Genetic Programming for Loss Function Learning (generalized form of [Liu et al., 2021](https://arxiv.org/abs/2102.04700) and [Li et al., 2021](https://arxiv.org/abs/2103.14026))

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

### Available Arguments:

- **Method** = {baseline, ml3, taylorglo, gplfl, evomal}
- **Dataset** = {mnist, cifar10, cifar100, svhn, california, boston, diabetes}
- **Model** = {linear, mlp, lenet5, alexnet, vgg, allcnnc, resnet, preresnet, wideresnet, squeezenet, pyramidnet}

### Code Reproducibility: 

The code has not been comprehensively checked and re-run since refactoring. If you're having any issues, find
a problem/bug or cannot reproduce similar results as the paper please [open an issue](https://github.com/Decadz/Evolved-Model-Agnostic-Loss/issues)
or email me.

## Reference

If you use our library or find our research of value please consider citing our papers with the following Bibtex entry:

```
@article{raymond2023learning,
  title={Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning},
  author={Raymond, Christian and Chen, Qi and Xue, Bing},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
@inproceedings{raymond2023fast,
  title={Fast and Efficient Local-Search for Genetic Programming Based Loss Function Learning},
  author={Raymond, Christian and Chen, Qi and Xue, Bing and Zhang, Mengjie},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={1184--1193},
  year={2023}
}
```
