# Optimization and classification of regularized mixtures of scaled Gaussian distributions, a.k.a regularized compound Gaussian distributions

![Build](https://github.com/antoinecollas/optim_compound/workflows/tests/badge.svg)

This repository hosts Python code for the numerical experiments of the the associated [arXiv paper](https://arxiv.org/abs/2209.03315).


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
@misc{collas22MSG,
      title = {Riemannian optimization for non-centered mixture of scaled Gaussian distributions},
      author = {Collas, Antoine and Breloy, Arnaud and Ren, Chengfang and Ginolhac, Guillaume and Ovarlez, Jean-Philippe},
      year = {2022},
      url = {https://arxiv.org/abs/2209.03315}
}
```
