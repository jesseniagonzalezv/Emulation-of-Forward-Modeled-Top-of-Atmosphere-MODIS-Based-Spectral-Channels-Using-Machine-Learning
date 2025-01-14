
import joblib
import pickle as pk
import tensorflow as tf
import sys
import xarray as xr
import pandas as pd
import os
import numpy as np
import math

from sklearn.base import BaseEstimator
from pickle import load
from sklearn.preprocessing import StandardScaler, MinMaxScaler




def load_dataframe_from_netcdf(path_data, prefix, fold_num):
    """
    Loads a DataFrame from a netCDF file based on the provided path, prefix, and fold number.

    Parameters:
    - path_data: str, the directory where the netCDF files are stored.
    - prefix: str, the prefix of the file to identify the type of data (e.g., 'df_icon_pca_train').
    - fold_num: int, the fold number for which the data should be loaded.

    Returns:
    - df: pandas.DataFrame, the DataFrame loaded from the netCDF file.
    """
    filename = f'{path_data}/{prefix}_fold_{fold_num}.nc'
    xr_dataset = xr.open_dataset(filename)
    df = xr_dataset.to_dataframe()

    return df


def save_model(model, path_output, name_model):  # scaler_x, pca_x, scaler_y,
    """
    The function saves the model in Joblib format

    Parameters:
        model (object): Trained machine learning model to be saved.
        path_output (str): Path to the directory where the model and PCA transformer will be saved.
        name_model (str): Name for the model file.

    Returns:
        None: The function doesn't return any value but saves the model to the specified path.

    Notes:
        The model is saved in '.joblib' format
    """

    # Check if the directory exists; if not, create it
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # Detect model type and save accordingly
    if isinstance(model, tf.keras.Model):
        model.save(os.path.join(path_output, name_model + ".h5"))
    elif isinstance(model, BaseEstimator):
        joblib.dump(model, os.path.join(path_output, name_model + ".joblib"))
    else:
        raise ValueError("Unsupported model type!")


