# Topological loss functions for deep metric learning problems

This repository is dedicated to my thesis work after the 4th year at HSE University. 

This code uses Python language, [PyTorch](https://pytorch.org) and [Open Metric Learning](https://github.com/OML-Team/open-metric-learning) libraries with all their dependecies. All requirements can be seen in `requirements.txt`.

All experiments are made on the following datasets 
- CUB-200-2011
- Stanford Online Products
- Deep Fashion

All runs were logged in [wandb](https://wandb.ai/nik-fedorov/topo_metric_learning).

## How to launch experiments

The main hyperparameters with all other configuration of training and evaluation process is organized into single configuration files handled by [hydra](https://hydra.cc/). 

Validation is done automatically during training every `valid_period` epochs (this parameter is also adjustable via configuration file). 

For launching baseline experiments your first need to activate your virtual environment `source <your_venv_folder>/bin/activate`, followed by installing all dependencies from `requirements.txt` file using command `pip install -r requirements.txt` and launching the training pipeline with command
```shell
train.py --config-name inshop_baseline0.yaml
```
Notes:
- it is possible to pass any other desired config from `configs` directory
- all configurations from file can be quickly and conveniently changed from command line (see [hydra overriding docs](https://hydra.cc/docs/advanced/override_grammar/basic/))
