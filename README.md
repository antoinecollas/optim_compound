# Optimization and classification of regularized mixtures of scaled Gaussian distributions, a.k.a regularized compound Gaussian distributions

![Build](https://github.com/antoinecollas/optim_compound/workflows/tests/badge.svg)

This repository hosts Python code for the numerical experiments of the the associated [coming soon...](https://arxiv.org/).


## Installation

The script `install.sh` creates a conda environment with everything needed to run the examples of this repo and installs the package:

```
./install.sh
```

## Check

To check the installation, activate the created conda environment `optim_compound` and run the unit tests:

```
conda activate optim_compound
nose2 -v --with-coverage
```


## Run numerical experiments

To run experiments, run the scripts from the different folders `center_of_mass/`, `classification/`, `estimation/` e.g.

```
python estimation/speed_comparison.py
```


## Cite

If you use this code please cite:

```
@misc{collas2022compound,
      title={...}, 
      author={...},
      year={2022},
      eprint={...},
      archivePrefix={arXiv},
      primaryClass={...}
}
```
