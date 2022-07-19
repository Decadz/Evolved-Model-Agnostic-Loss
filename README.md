# Evolved Model-Agnostic Loss

This repository contains code to reproduce the experiments in the paper 
"*Learning Symbolic Model-Agnostic Loss Functions via Meta-Learning*" by 
Christian Raymond, Qi Chen, Bing Xue and Mengjie Zhang.

## Contents

This repository contains an implementation of the proposed Evolved Model Agnostic Loss (EvoMAL) algorithm. 
In addition, it contains code for reproducing the following loss function learning algorithms:

* Meta-Learning via Learned Loss (ML<sup>3</sup>) Supervised
* Taylor Genetic Loss Optimization (TaylorGLO)
* Genetic Programming for Loss Function Learning (GP-LFL)

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

- method_name = {baseline, ml3, taylorglo, gplfl, evomal}
- dataset_name = {mnist, cifar10, cifar100, svhn}
- model_name = {lenet5, alexnet, vgg, allcnnc, resnet, preresnet, wideresnet, squeezenet, pyramidnet}
  
### Additional Notes: 

* In addition to the compulsory arguments all hyper-parameters (found in the 'configurations') as well as the *device* can be given as optional arguments which will override the default values.
* The 'run_landscapes.py' script generates loss landscapes which can be either visualized directly in PyVista or exported to a '.vtk' file which can be reopened in the open sources software ParaView.
* The meta-learned loss function and/or trained model used as a source for 'run_transfer.py' or 'run_landscapes.py' should be placed in the respective file in 'experiments/results/' in order for code to work as intended.

### Code Reproducibility: 

The code have not been comprehensively checked and re-run since refactoring. If you're having any issues or find a problem/bug or cannot reproduce similar results as the paper please open an issue or email me.

## Reference

