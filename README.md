# Taarruz
Adversarial Attack Tool for [Knet](github.com/denizyuret/Knet.jl).

## Implemented Attacks
- ```FGSM```: Fast Gradient Sign Method ([paper](arxiv.org/abs/1412.6572)).

Too see documentation, type ```@doc method_name``` in Julia REPL (e.g. ```@doc FGSM```).

## Example Notebooks
- [lenet-fgsm](examples/lenet-fgsm.ipynb): FGSM attack to Lenet trained on MNIST dataset.
