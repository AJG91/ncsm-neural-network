# NCSM Neural Network

This repository contains all the code and data necessary to train an feed-forward artificial neural network (ANN) to predict ground-state properties of He-4 using [no-core shell model (NCSM)][Bogner] data.
This work is based on the following papers: [ANN][Negoita], [custom loss][Jiang].

## Getting Started

* This project relies on `python=3.9`. It was not tested with different versions.
  To view the entire list of required packages, see `environment.yml`.
* Clone the repository to your local machine.
* Once you have `cd` into this repo, create a virtual environment (assuming you have `conda` installed) via
```bash
conda env create -f environment.yml
```
* Enter the virtual environment with `conda activate ncsm-neural-network-env`
* Install the `libraries` package in the repo root directory using `pip install -e .`
  (you only need the `-e` option if you intend to edit the source code in `libraries/`).

## Example

The main class for the neural network is `NeuralNetwork`, which implements the neural network.
The code snippet below shows how it should be used:
```python
from libraries import NeuralNetwork

# Setup
```

```

[Bogner]: https://www.sciencedirect.com/science/article/abs/pii/S0375947407008147?via%3Dihub
[Negoita]: https://arxiv.org/abs/1803.03215
[Jiang]: https://journals.aps.org/prc/abstract/10.1103/PhysRevC.100.054326
