# Soil spectroscopy toolbox

## General instructions

1. Load the data. This is any set of numerical features (X) and labels (Y), structured as pandas DataFrames and divided into a training and test set.
2. Instantiate an Analyzer, or a subclass of one.
3. Train the Analyzer on the training data.
4. Evaluate the Analyzer using the test set.


## Input data

A dataset should be a pandas DataFrame with:

- A column for soil organic carbon (SoC) in % by weight
- A column for electroconductivity (EC)
- A column or columns for sample depth in cm
- Latitude
- Longitude
- All remaining columns are reflecivity/spectral data in nm

## Analyzers

This repository contains a number of spectroscopy calibration models that inherit from the `Analyzer` class. To the greatest extent possible, Analyzers are treated as black boxes: soil features go in, predictions come out.

### Planned models

- [x] Multi-layer perceptron (MLP)
- [x] Random Forest (RF)
- [x] Partial Least Squares Regressor (PLSR)
- [ ] Transformer-based
- [ ] Hybrid LSTM and CNN model (see [Wang et al](https://www.sciencedirect.com/science/article/pii/S016816992300738X?entityID=https%3A%2F%2Flogin.cmu.edu%2Fidp%2Fshibboleth&pes=vor))

## Quirks

### Tensorflow can't find your GPU

First ensure you've met the hardware and system requirements outlined in the [installation instructions](https://www.tensorflow.org/install/pip), then try [this](https://stackoverflow.com/a/77528450/6238455):

```bash
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib
```
