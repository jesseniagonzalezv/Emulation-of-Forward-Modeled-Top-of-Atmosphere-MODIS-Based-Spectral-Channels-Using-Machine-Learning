 
import numpy as np
import pandas as pd
import tensorflow as tf

from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential

def rmse_2(y_true, y_pred):
    # Calculate RMSE for each output
    rmse_per_output = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred), axis=0))
    
    # Aggregate RMSEs, here we use the mean RMSE across all outputs
    mean_rmse = tf.reduce_mean(rmse_per_output)

    return mean_rmse
    
# RMSE metric
# # Define R^2 metric
def r2_score_2(y_true, y_pred):
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res/(ss_tot + tf.keras.backend.epsilon())



