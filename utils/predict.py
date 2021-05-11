import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow import keras
from loguru import logger

model_path = f'results/best_model.hdf5'
data_path = f'results/clean_data.csv'
output_folder = f'results/'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def prepare_data_to_predict(time_data):

    return time_data.values.reshape((time_data.shape[0], time_data.shape[1], 1))

def predict_labels(time_data, model_path):
    # evaluate best model on new dataset
    x_predict = prepare_data_to_predict(time_data)
    model = keras.models.load_model(model_path)
    y_pred = model.predict(x_predict)
    y_pred = np.argmax(y_pred, axis=1)

    # Add labels back to original dataframe
    time_data['label'] = y_pred

    return time_data

if __name__ == '__main__':

    # Read in dataset
    raw_data = pd.read_csv(data_path)
    raw_data.drop([col for col in raw_data.columns.tolist() if 'Unnamed: ' in col], axis=1, inplace=True)

    # prepare time series data
    time_data = raw_data[[col for col in raw_data.columns.tolist() if col not in ['molecule_number', 'label']]].T.reset_index(drop=True)

    time_data = predict_labels(time_data, model_path)

    # Save to csv
    raw_data.to_csv(f'{output_folder}predicted_labels.csv')
