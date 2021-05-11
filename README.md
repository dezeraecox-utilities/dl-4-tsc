# Deep Learning for Time Series Classification

This is a fork of the original repository from [Fawaz et al 2019](https://link.springer.com/article/10.1007%2Fs10618-019-00619-1) titled "Deep learning for time series classification: a review" published in [Data Mining and Knowledge Discovery](https://link.springer.com/journal/10618), also available on [ArXiv](https://arxiv.org/pdf/1809.04356.pdf).

## Why create a fork?

This repository was forked for incorporation into an analysis pipeline designed to process single molecule fluorescence data. Code edits were made to enable this module to be installed via PyPi and thus expose the relevant functions for applying any classifier to a labelled training DataFrame. Additional functions were added to enable the best model to then be loaded and used for prediction of additional novel datasets.

## Code

The code is divided as follows:

- The [main.py](main.py) python file contains the necessary code to train a new model against a benchmarking dataset. See below for requirements and input format.
- The [utils](utils) folder contains the necessary functions to read the datasets and visualize the plots used in the initial paper. These functions have been left untouched. Additional functions to apply pre-trained models for predictive purposes have also been added.
- The [classifiers](classifiers) folder contains nine python files, one for each deep neural network tested in the original paper. These files, as far as possible, have been left untouched.

## Prerequisites

All required python packages are listed in the [environment.yml](environment.yml) file. It is recommended to use Conda for environment management, and a new conda environment containing all the dependencies can be created at the command line using:

```conda env create -f environment.yml```

The code currently uses Tensorflow 2.0, and will likely be minimally maintained inline with forthcoming versions. In addition, there are several version inconsistencies between tensorflow and hdf5/h5py, meaning the versions listed in the environment.yml file may be incompatible at some stage.

## Getting started

### Train a model

To train a model against a new dataset, you first need a DataFrame containing unique sample identifiers mapped to the corresponding class label. After labelling, the input dataset should contain a 'label' column in which the class is specified and the timeseries datapoints occupy the remainder of the columns:

|Sample   | Label   | t0  | t1  | t2  |
|---------|---------|-----|-----|-----|
| 1       |0        | 51  |68   |91   |
| 2       |1        | 31  |20   |10   |
| 3       |2        | 40  |40   |41   |
| 4       |0        | 45  |70   |85   |
| 5       |1        | 45  |31   |16   |

Once this data is prepared, it should then be split into timepoint data and label data which is then compatible with the fit_new_model function found in the [main](main.py) script:

```
from dl4tsc.main import train_new_model

# define location to save model
output_folder = 'results/'

# prepare time series data
time_data = raw_data[[col for col in raw_data.columns.tolist() if col in timepoints]].T.reset_index(drop=True)

# prepare label data
labels = raw_data['label'].copy().astype(int)

# train model
train_new_model(time_data, labels, output_folder, classifier_name='resnet')

```
### Predict labels for a new dataset

Once you have a complete model trained (and are satisfied with the training metrics), then this model can be reloaded at any time for prediction of new datasets. Functions for this are provided in the [predict.py](predict.py) script, and can be used on data as shown above (obviously in the absence of the label column). 
```
from dl4tsc.utils.predict import predict_labels

labelled_data = predict_labels(time_data, model_path='results/best_model.hdf5')

```

## Reference

If you re-use this work, please cite the original paper:

```
@article{IsmailFawaz2018deep,
  Title                    = {Deep learning for time series classification: a review},
  Author                   = {Ismail Fawaz, Hassan and Forestier, Germain and Weber, Jonathan and Idoumghar, Lhassane and Muller, Pierre-Alain},
  journal                  = {Data Mining and Knowledge Discovery},
  Year                     = {2019},
  volume                   = {33},
  number                   = {4},
  pages                    = {917--963},
}
```