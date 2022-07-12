# Junction Tree Variational Autoencoder for Molecular Graph Generation

Official implementation of Junction Tree Variational Autoencoder [https://arxiv.org/abs/1802.04364](https://arxiv.org/abs/1802.04364)
This repo will use to conduct de novo drug design by multi-objective optimization.

# Update
This repo contains Python script (version 3.6.13) that be able to train JTVAE model.

# Requirements
* Linux (We only tested on Ubuntu)
* RDKit (version == 2017.09)
* Python (version == 3.6.13)
* PyTorch (version == 1.10.2)
* Theano (version in GrammarVAE)

To install RDKit, please follow the instructions here [http://www.rdkit.org/docs/Install.html](http://www.rdkit.org/docs/Install.html)

We highly recommend you to use conda for package management.

The following directories provides scripts for the experiments in original ICML paper:
* `bo/` includes scripts for Bayesian optimization experiments. Please read `bo/README.md` for details.
* `molvae/` includes scripts for training our VAE model only. Please read `molvae/README.md` for training VAE model.
* `molopt/` includes scripts for jointly training our VAE and property predictors. Please read `molopt/README.md` for details.
* `jtnn/` contains codes for model formulation.
