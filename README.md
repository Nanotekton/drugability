# Drugability - Machine Learning Study

Here we deposit the code associated with the article *Advantages and limitations of Artificial Intelligence in defining general drug-likeness*
by Wiktor Beker, Agnieszka Wołos and Bartosz A. Grzybowski.

## Prerequisites

Before start, please make sure to have installed following packages (version used in this particular project indicated in parentheses):

* [RdKit](http://www.rdkit.org/) (version  2017.09.1)
* [Keras](https://keras.io/) (version 2.2.4 with TensorFlow backend)
* [Keras Deep Learning on Graphs](https://vermamachinelearning.github.io/keras-deep-graph-learning/)
* [Sci-kit learn](https://scikit-learn.org) (version 0.21.2)
* [Imbalanced-learn](https://imbalanced-learn.org) (version 0.4.3)
* [Mol2Vec](https://mol2vec.readthedocs.io) (version 0.1)
* [PyYAML](https://pyyaml.org/) (version 5.2)

## Code organisation

The general idea was to separate common steps in machine learning experiments (vectorisation, model training and testing) into logical components and to control the configuration of ML experiment with YAML files, making room for non-standard procedures (that is such not already implemented in Sklearn).
As suggested by name, ```data_preprocess.py``` contains functions responsible for data loading and preprocess, whereas ```models.py``` collects functions constructing a model according to a given config. 
The ```make_experiment.py``` script governs the overall workflow according to a given YAML config file (see ```python make_experiment.py -h``` and examples provided below).

PU-learning scripts are placed in separate directory (```scripts/pu_fuselier.py``` for the Fuselier method and ```scripts/pu_iteration.py``` for the Liu approach) together with some additional tools.

### Configuration file

In general, configuration file is comprised of three parts: ```loader_config```, ```model_params``` and ```cross_validation``` (in some cases additional ```vectorizer_config``` part may be passed). The names are meant to be self-explanatory; see ```config_files/``` for examples.

In ```loader_config```, position ```data_balance``` controls data balancing:
* ```weights``` computes sample weight according to the size of each class,
* ```random``` selects random subset of the majority class (of the same size as the minority class),
* ```cut``` selects first part of the majority class (of the same size as the minority class).

## Examples
### RdKit and Mol2Vec vectorization

For RdKit and Mol2Vec representations, the data has to be vectorized first (the names used below correspond to paths included in the config files).
Note that with Mol2Vec, you should provide path to pickle with Mol2Vec model (not included in the repo).
```
python scripts/vectorize.py --output_core data/drugs_approved_rdkit --descriptor rdkit  data/drugs_approved.smiles
python scripts/vectorize.py --output_core data/drugs_approved_m2v --descriptor mol2vec --mol2vec_model_pkl MOL2VEC_MODEL_PKL  data/drugs_approved.smiles
python scripts/vectorize.py --output_core data/zinc15_nondrugs_sample_rdkit --descriptor rdkit  data/zinc15_nondrugs_sample.smiles
python scripts/vectorize.py --output_core data/zinc15_nondrugs_sample_m2v --descriptor mol2vec --mol2vec_model_pkl MOL2VEC_MODEL_PKL  data/zinc15_nondrugs_sample.smiles
```
### Sample perceptrons
#### RdKit
```
```
### PU-learning example
### Multi-task learning example
### Balanced Random Forest example 

## Closing remarks
This repository was created as a part of scientific research rather than a standalone project. Due to tight schedule, it's quite 'sketchy'. I hope that with incoming projects I manage to transform it into a flexible framework for ML experiments with a possibility to have a lower-level control over each step (eg. user-defined splitting of the data). 

### ToDo
* unify Keras and Sklearn APIs in ```make_experiment.py```
