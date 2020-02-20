# Drugability - Machine Learning Study

Here we deposit the code associated with the article *Advantages and limitations of Artificial Intelligence in defining general drug-likeness*
by Wiktor Beker, Agnieszka Wołos and Bartosz A. Grzybowski.

## Prerequisites

Before start, please make sure to have installed following packages (version used in this particular project indicated in parentheses):

* [RdKit](http://www.rdkit.org/) (version  2017.09.1)
* [Keras](https://keras.io/) (version 2.2.4 with TensorFlow backend)
* [Keras Deep Learning on Graphs](https://vermamachinelearning.github.io/keras-deep-graph-learning/)
* [Sci-ki learn](https://scikit-learn.org) (version 0.21.2)
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

## Examples
```
Under construction
```
### Sample perceptrons
### PU-learning example
### Multi-task learning example
### Balanced Random Forest example 

## Closing remarks
This repository was created as a part of scientific reasearch rather than a standalone project. Due to tight schedule, it's quite 'sketchy'. I hope that with incoming projects I manage to transform it into a flexible framework for ML experiments with a possibility to have a lower-level control over each step (eg. user-defined splitting of the data). 

### ToDo
* unify Keras and Sklearn APIs in ```make_experiment.py```
