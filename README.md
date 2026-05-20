# FEM

`fem-partition` is a python library for solving graph partition problems using `FEM` framework. The current build-in problem types are:
* normal graph balance minimum cut
* hypergraph balance minimum cut

## Installation

1. One can use conda to install the package with the following commands:
    ```bash
    conda env create -f environment.yml
    ```
    this will create an environment named `fem` with all the dependencies except for the pytorch, then activate the environment with `conda activate fem`.

2. Then `pytorch` have to be installed manually with 
    ```bash
    pip3 install torch torchvision torchaudio
    ```
    see the [pytorch website](https://pytorch.org/) for more details.
