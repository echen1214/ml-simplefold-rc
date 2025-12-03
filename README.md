
<h1 align="center"><strong>RCFold_Regressor</strong></h1>


<div align="center">

This github repository is a fork of the [repo](https://github.com/apple/ml-simplefold) corresponding of the SimpleFold research paper, [*SimpleFold: Folding Proteins is Simpler than You Think*](https://arxiv.org/abs/2509.18480). This repo develops a model to perform in silico supervised learning for the [2023 AlignBio Protein Engineering Tournament](https://github.com/Align-to-Innovate/the-protein-engineering-tournament-2023)

</div>

## Introduction

This educational project was born out of the F2'25 batch of the [Recurse Center](https://www.recurse.com/). Originally, the aim of the project was to develop a version analogous to the work done in this [paper](https://www.mlsb.io/papers_2024/Low-N_OpenFold_fine-tuning_improves_peptide_design_without_additional_structures.pdf). However, the scope has since changed to retrospectively compete in the [2023 AlignBio Protein Engineering Tournament](https://github.com/Align-to-Innovate/the-protein-engineering-tournament-2023).

## Results
The main thrusts of the project were to
- Collaboratively bootstrap model development from scratch
- Explore Hydra configs, Pytorch lightning, and Wandb following best practices described in this [repo](https://github.com/ashleve/lightning-hydra-template). This workflows allows organization of experiments and logging,  scalability across to multiple datasets, and collaborative do model development (including a null-hypothesis model) 
- Perform hyperparameter search with Wandb
- Provide data pre-processing pipelines and dataset formats tailored to the protein engineering benchmarks.
- Add regression heads and training scripts to predict various metrics from the ESM protein language features
- Perform exploratory data and embedding space analysis with t-SNE and Marimo

On-going:
- Explore XGboosting model development and LoRA fine-tuning of ESM
- Curate proper training and test set evaluations. For the alpha-amylase dataset there appears to be a lot of training set leakage. 
- Incorporate the GPT-2 baseline 
- Perform CI/CD workflow

Potential research ideas:
- Explore how ESM positional encoding works
- Use latent space of SimpleFold model as feature embedding for the regression modules for the tasks with fewer datapoints

</div>


## Installation

To install `simplefold` package from github repository, run
```
git clone https://github.com/apple/ml-simplefold.git
cd ml-simplefold
conda create -n simplefold python=3.10
python -m pip install -U pip build; pip install -e .
```
If you want to use MLX backend on Apple silicon: 
```
pip install mlx==0.28.0
pip install git+https://github.com/facebookresearch/esm.git
```
There are also additional depedencies to visualize, train, and run the various regression models
```
pip install uv marimo
pip install peft transformers
pip install xgboost
```

## Acknowledgements
Our codebase is built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details. 

## License
Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.
