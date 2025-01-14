
import numpy as np
import pandas as pd
import pickle
import argparse
import tensorflow as tf

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam

from metric_utils import metric_calculation, r2_score_2, rmse_2
from data_utils import load_dataframe_from_netcdf, save_model
from plotting_utils import plot_loss_train_val
from utils import format_times
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- FUNCTIONS --------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

def model_config(type_model, lr, n_epochs, x_train, y_train, x_val, y_val, batch_size):
    """
    Configure a neural network model based on the specified type.

    This function builds a Sequential model based on the type of model specified by 'type_model'.
    Currently, it supports several predefined neural network and random forest configurations

    Parameters:
        type_model (str): Type of the model to be created. 
        lr (float): Learning rate for the optimizer.
        n_epochs (int): Number of epochs for training the model.
        x_train (pandas.DataFrame): Training data.
        y_train (pandas.DataFrame): Target values for the training data.
        x_val (pandas.DataFrame): Validation data.
        y_val (pandas.DataFrame): Target values for the validation data.
        batch_size (int): Batch size for the training data
    Returns:
        model: A model configured based on the provided 'type_model'.

    """
    if type_model == "NN11":
        print(f" -------------- MODEL NN_type {type_model} -------------- ")
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(y_train.shape[1]))

        model.compile(optimizer=Adam(learning_rate=lr),  # 0.0001
                      loss='mean_squared_error', metrics=[r2_score_2, rmse_2])

        # early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            x_train, 
            y_train, 
            epochs=n_epochs, 
            verbose=2,
            validation_data=(x_val, y_val),
            batch_size=batch_size
        )

    elif type_model == "NN3":
        print(f" -------------- MODEL NN_type {type_model} -------------- ")
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
        model.add(Dropout(0.3))

        model.add(Dense(128, activation='relu'))
        
        model.add(Dense(64, activation='relu'))
        
        model.add(Dense(y_train.shape[1]))

        model.compile(optimizer=Adam(learning_rate=lr),  # 0.0001
                      loss='mean_squared_error', metrics=[r2_score_2, rmse_2])

        # early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        history = model.fit(
            x_train, 
            y_train, 
            epochs=n_epochs, 
            verbose=2,
            validation_data=(x_val, y_val),
            batch_size=batch_size
        )
    elif type_model == "NN4":
        print(f" -------------- MODEL NN_type {type_model} -------------- ")
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=(x_train.shape[1],)))
        model.add(Dropout(0.3))

        model.add(Dense(128, activation='relu'))
        
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(y_train.shape[1]))

        model.compile(optimizer=Adam(learning_rate=lr),  # 0.0001
                      loss='mean_squared_error', metrics=[r2_score_2, rmse_2])

        # early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        history = model.fit(
            x_train, 
            y_train, 
            epochs=n_epochs, 
            verbose=2,
            validation_data=(x_val, y_val),
            batch_size=batch_size
        )

    elif type_model == "CNN_op2":
        print(" -------------- MODEL CNN_op2 -------------- ")

        # Ensure the data is reshaped appropriately for Conv1D
        if len(x_train.shape) == 2:
            x_train = np.expand_dims(x_train, axis=-1)
        if len(x_val.shape) == 2:
            x_val = np.expand_dims(x_val, axis=-1)

        model = Sequential()

        # Convolutional Layers
        model.add(Conv1D(32, 3, activation='relu', input_shape=(x_train.shape[1], 1)))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))

        model.add(Conv1D(64, 3, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.2))

        # Flatten the data for the dense layers
        model.add(Flatten())

        # Dense Layers
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
                  
        model.add(Dense(y_train.shape[1]))

        model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error', metrics=[r2_score_2, rmse_2])
        # early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

        history = model.fit(
            x_train, 
            y_train, 
            epochs=n_epochs, 
            verbose=2,
            validation_data=(x_val, y_val),
            batch_size=batch_size
            # ,
            # callbacks=[early_stopping]
        )

    elif type_model == "RF2":
        
        print(" -------------- MODEL RF_type 2 -------------- ")
        rf_model = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                                         max_depth=30, max_features='auto', max_leaf_nodes=None,
                                         # max_depth=None,
                                         max_samples=None, min_impurity_decrease=0.0,
                                         # min_impurity_split=None, min_samples_leaf=1,
                                         min_samples_leaf=6,
                                         min_samples_split=2, min_weight_fraction_leaf=0.0,
                                         n_estimators=70, n_jobs=32, oob_score=False,
                                         random_state=42, verbose=1, warm_start=False)

        model = rf_model.fit(x_train, y_train)
        history = []
        
    elif type_model == "RF3":
        # max_depth=40, max_features=1, 
        
        print(" -------------- MODEL RF_type 3 -------------- ")
        rf_model = MultiOutputRegressor(RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='squared_error',
                                         max_depth=30, max_features=3, max_leaf_nodes=None,
                                         # max_depth=None,
                                         max_samples=None, min_impurity_decrease=0.0,
                                    
                                         min_samples_leaf=6,
                                         min_samples_split=2, min_weight_fraction_leaf=0.0,
                                         n_estimators=40, n_jobs=64, oob_score=False,
                                         random_state=42, verbose=1, warm_start=False))

        model = rf_model.fit(x_train, y_train)
        history = []
        
        
    return history, model


def print_configuration(args):
    configuration = f"""
    Configuration:
    - Model Type: {args.type_model}
    - Learning Rate: {args.lr}
    - Number of Epochs: {args.n_epochs}
    - Batch size: {args.batch_size}
    - Channel Number List: {args.channel_number_list}
    - Fold Number: {args.fold_num}
    - Dataser Path: {args.path_dataframes_pca_scaler}
    - Output Path: {args.path_models}
    """
    print(configuration)
    # - All Channels Joined: {args.all_chan_join}

# --------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- MAIN CODE --------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def main():
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data-type', type=str, default='ref_rad_total', help='name of the variable in the dataframe one')
    arg('--type-model', type=str, default='NN', help='select the model NN, RF')
    arg('--path_dataframes_pca_scaler', type=str, default="/data/", help='path where is the dataset save as dataframes and pca, scaler')
    arg('--path_models', type=str, default="output/", help='path of the folder to save the outputs')
    arg('--fold-num', type=int, default=1, help='n k-fold to run')  
    arg('--lr', type=float, default=1e-3)
    arg('--n-epochs', type=int, default=30)
    arg('--batch-size', type=int, default=64)
    arg('--channel_number_list', nargs='+', type=int, help='List of the channels to evaluate (no index)')
    # arg('--all_chan_join', type=str, default="true", help='True false to train all the channel of channel number list in the same time')

    args = parser.parse_args()

    type_model = args.type_model
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    channel_number_list = args.channel_number_list
    fold_num = args.fold_num
    data_type = args.data_type
    path_dataframes_pca_scaler = args.path_dataframes_pca_scaler
    path_models = args.path_models

    print(" ========================================== ")
    print_configuration(args)
    # Read back the data
    with open(f"{path_dataframes_pca_scaler}/times_data_k_fold_{fold_num}.pkl", 'rb') as f:
        times_data = pickle.load(f)
    with open(f"{path_dataframes_pca_scaler}/lat_lon_data_{fold_num}.pkl", 'rb') as f:
        lat_lon_data = pickle.load(f)
    df_icon_pca_train = load_dataframe_from_netcdf(path_dataframes_pca_scaler, 'df_icon_pca_train', fold_num)
    df_icon_pca_val = load_dataframe_from_netcdf(path_dataframes_pca_scaler, 'df_icon_pca_val', fold_num)
    df_icon_pca_test = load_dataframe_from_netcdf(path_dataframes_pca_scaler, 'df_icon_pca_test', fold_num)
    df_ref_train = load_dataframe_from_netcdf(path_dataframes_pca_scaler, 'df_ref_train', fold_num)
    df_ref_val = load_dataframe_from_netcdf(path_dataframes_pca_scaler, 'df_ref_val', fold_num)
    df_ref_test = load_dataframe_from_netcdf(path_dataframes_pca_scaler, 'df_ref_test', fold_num)
    
    name_saving_files = f"{type_model}_k_fold_{fold_num}"

    channel_number_list_idx = np.array([x - 1 for x in channel_number_list])
    print(f"--------- indexes channels: {channel_number_list_idx}")
    channel_groups = []
    channel_groups = [(channel_number_list_idx, data_type)] 

    for channels, name in channel_groups:
        print(f"-----------------training model to {name} ---------------------")
            
        history, model = model_config(type_model=f"{type_model}",
                                      lr=lr,
                                      n_epochs=n_epochs,
                                      x_train=df_icon_pca_train,
                                      y_train=df_ref_train.iloc[:, channels],
                                      x_val=df_icon_pca_val, 
                                      y_val=df_ref_val.iloc[:, channels],
                                      batch_size=batch_size)  # 128 only I got 0.5 with 64 it is better

        save_model(model=model,
                   path_output=path_models,
                   name_model=f"{name}_{name_saving_files}")

        # ========== evaluate loss test ================
        if type_model != "RF" and type_model != "RF2" and type_model != "RF3":
            # test_loss = model.evaluate(df_icon_pca_test, df_ref_test.iloc[:, channels], verbose=0)
            # print(f"Test loss: {test_loss}")

            plot_loss_train_val(history=history,
                                type_model=f"{name}_{name_saving_files}",
                                path_output=path_models)

        # ========== Evaluate your model with x_test, y_test ==========
        # Calculate metrics
        metrics_train, _ = metric_calculation(x=df_icon_pca_train,
                                              gt=df_ref_train.iloc[:, channels],
                                              model_ml=model,
                                              data_name="training")

        metrics_val, _ = metric_calculation(x=df_icon_pca_val,
                                            gt=df_ref_val.iloc[:, channels],
                                            model_ml=model,
                                            data_name="validation")

        metrics_test, test_predictions_ref_rad = metric_calculation(x=df_icon_pca_test,
                                                                    gt=df_ref_test.iloc[:, channels],
                                                                    model_ml=model,
                                                                    data_name="testing")

        train_times = format_times(times_data['train_times'])
        val_times = format_times(times_data['val_times'])
        test_times = format_times(times_data['test_times'])
        
        # Modified print statements
        print(f"Training times for k={fold_num}: {train_times}")
        print(f"Validation times for k={fold_num}: {val_times}")
        print(f"Testing times for k={fold_num}: {test_times}")

        flatted_size_img = len(lat_lon_data['lat']) * len(lat_lon_data['lon'])
        # len(y_test.lat) * len(y_test.lon)
        
        # If I deleted the some pixels in the test data then the next code is not possible to apply to divide the 3 timesteps
        for i, n_time in enumerate(times_data['test_times']):
            # (y_test.time.values):
            _, test_predictions_ref_rad = metric_calculation(x=df_icon_pca_test.reset_index(drop=True)[i*flatted_size_img:(i+1)*flatted_size_img],
                                                             gt=df_ref_test.iloc[:, channels].reset_index(drop=True)[i*flatted_size_img:(i+1)*flatted_size_img],
                                                             model_ml=model,
                                                             data_name=f"testing in T{n_time}")
            
        # ========== Create DataFrame ==========
        df_metrics = pd.DataFrame({
            f"Train k={fold_num} ({train_times})": metrics_train,
            f"Validation ({val_times})": metrics_val,
            f"Test ({test_times})": metrics_test,
        }).T

        df_metrics = df_metrics.round(4)
        df_metrics.to_csv(f"{path_models}/table_metrics_{name_saving_files}_{name}_with_scaler.csv")
        print(df_metrics)


if __name__ == '__main__':
    main()
