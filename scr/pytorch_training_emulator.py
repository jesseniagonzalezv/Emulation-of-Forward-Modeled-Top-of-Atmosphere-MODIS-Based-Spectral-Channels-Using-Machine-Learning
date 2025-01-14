
import numpy as np
import pandas as pd
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# from metric_utils import metric_calculation, r2_score_2, rmse_2
from data_utils import load_dataframe_from_netcdf
# from plotting_utils import plot_loss_train_val
from utils import format_times
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

def compute_rmse(y_pred, y_true):
    # return sqrt(mean_squared_error(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()))

    # Ensure y_pred and y_true are on the same device
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    # Calculate MSE for each output
    mse_per_output = np.mean((y_true - y_pred) ** 2, axis=0)
    
    # Take the square root of the MSE to get RMSE for each output
    rmse_per_output = np.sqrt(mse_per_output)
    
    # Average the RMSEs across all outputs
    mean_rmse = np.mean(rmse_per_output)
    
    return mean_rmse


def compute_r2(y_pred, y_true):
    return r2_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())  # multioutput='uniform_average' (defaults)

# --------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- FUNCTIONS --------------------------------------------------
# --------------------------------------------------------------------

def metric_calculation_all_data(outputs, targets, data_name, raw_values=False):
    """
    Evaluate R-squared (R2) and Root Mean Squared Error (RMSE) for each output of the model and their uniform averages.

    Parameters:
    - outputs (torch.Tensor): The model's predictions, expected to be a tensor of shape (n_samples, n_outputs).
    - targets (torch.Tensor): The ground truths (actual values), expected to be a tensor of shape (n_samples, n_outputs).
    - raw_values (bool): Whether to include raw values for each output channel.

    Returns:
    - metrics (dict): A dictionary containing RMSE and R2 for all output in averaged and optionally for each channel.
    """
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
   
    metrics = {
            # "Data Name": data_name,
        }

    # Calculate range (max - min) of target
    range_per_output = np.ptp(targets, axis=0)  # ptp = peak to peak 

    if raw_values:
        rmse_metrics = {}
        r2_metrics = {}
        mae_metrics = {}
        mape_metrics = {}
        nrmse_metrics = {}
    
        rmse_raw = np.sqrt(mean_squared_error(targets, outputs, multioutput='raw_values'))
        r2_raw = r2_score(targets, outputs, multioutput='raw_values')
        nrmse_raw = (rmse_raw / range_per_output) * 100
        mae_raw = mean_absolute_error(targets, outputs, multioutput='raw_values')
        mape_raw = np.mean(np.abs((targets - outputs) / targets), axis=0) * 100

        metrics = {
                    'Channel': [],
                    'nRMSE': [],
                    'R2': [],
                    'RMSE': [],
                    'MAE': [],
                    'MAPE': []
                }
        for i in range(len(rmse_raw)):

            metrics['Channel'].append(f"{i+1}")
            metrics['RMSE'].append(rmse_raw[i])
            metrics['R2'].append(r2_raw[i])
            metrics['MAE'].append(mae_raw[i])
            metrics['MAPE'].append(mape_raw[i])
            metrics['nRMSE'].append(nrmse_raw[i])            

    else:
        # Calculate RMSE for each output
        rmse_per_output = np.sqrt(np.mean((targets - outputs) ** 2, axis=0))
    
        # Calcular NRMSE por cada salida
        nrmse_per_output = rmse_per_output / range_per_output
    
        # Calculate the mean RMSE across all outputs
        rmse_avg = np.mean(rmse_per_output) 
    
        nrmse_avg = np.mean(nrmse_per_output) * 100
    
        r2_avg = r2_score(targets, outputs, multioutput='uniform_average')
        mae_avg = mean_absolute_error(targets, outputs, multioutput='uniform_average')
        mape_avg = np.mean(np.abs((targets - outputs) / targets)) * 100
    
        metrics = {
            "R-squared (R2)": r2_avg,
            "Root Mean Squared Error (RMSE)":rmse_avg,
            "Normalized Mean Squared Error (nRMSE %)":nrmse_avg,
            "Mean Absolute Error (MAE)":mae_avg,
            "Mean Absolute Error (MAPE %)":mae_avg,
        }
        print(f"------------------ {data_name} metrics ----------------------")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")    

    return metrics
    

def calculate_rmse_r2(outputs, targets):
    outputs = outputs.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    # rmse = np.sqrt(mean_squared_error(targets, outputs, multioutput='uniform_average'))
    # Calculate RMSE for each output
    rmse_per_output = np.sqrt(np.mean((targets - outputs) ** 2, axis=0))
    # Calculate the mean RMSE across all outputs
    rmse = np.mean(rmse_per_output)
    r2 = r2_score(targets, outputs, multioutput='uniform_average')
    return rmse, r2
    
def metric_evaluation_each_output(outputs, targets, channels_idx):
    """
    Evaluate R-squared (R2) and Root Mean Squared Error (RMSE) for each output of the model and their uniform averages.

    Parameters:
    - outputs (torch.Tensor): The model's predictions, expected to be a tensor of shape (n_samples, n_outputs).
    - targets (torch.Tensor): The ground truths (actual values), expected to be a tensor of shape (n_samples, n_outputs).

    Returns:
    - metrics (dict): A dictionary containing RMSE and R2 for each output individually.
    """
    outputs = outputs.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    rmse_metrics = {}
    r2_metrics = {}
    
    for i, chan in range(channels_idx):
    # outputs.shape[1]):  # Iterate through each output (column)
        rmse = np.sqrt(mean_squared_error(targets[:, i], outputs[:, i]))
        r2 = r2_score(targets[:, i], outputs[:, i])
        rmse_metrics[chan + 1] = rmse
        r2_metrics[chan + 1] = r2


    return rmse_metrics, r2_metrics



def plot_loss(train_losses, val_losses, path_output, type_model):
    f = plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Save the figure
    figure_name = f'{path_output}/history_{type_model}.png'
    f.tight_layout()
    f.savefig(figure_name, dpi=60)
    plt.show()


# def model_config(type_model, lr, n_epochs, x_train, y_train, x_val, y_val, batch_size):
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

class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)
    

    def set_sizes_from_state_dict(self, state_dict):
        # Automatically infer input and output sizes from the state_dict
        for name, param in state_dict.items():
            if '0.weight' in name:
                self.input_size = param.shape[1]
            elif '8.weight' in name:
                self.output_size = param.shape[0]



def train_model(model, train_loader, val_loader, n_epochs, optimizer, criterion, device):
    best_val_rmse = float('inf')
    train_losses, val_losses = [], []
    train_rmses, val_rmses = [], []
    train_r2s, val_r2s = [], []

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_targets = []

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_train_loss += loss.item()
            all_train_preds.append(outputs)
            all_train_targets.append(targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = total_train_loss / len(train_loader)
        
        all_train_preds = torch.cat(all_train_preds)
        all_train_targets = torch.cat(all_train_targets)
        train_rmse = compute_rmse(all_train_preds, all_train_targets)
        train_r2 = compute_r2(all_train_preds, all_train_targets)

        train_losses.append(train_loss)
        train_rmses.append(train_rmse)
        train_r2s.append(train_r2)

        # Validation phase
        val_loss, val_rmse, val_r2, all_preds, all_targets = evaluate_model(model, val_loader, criterion, device)

        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        val_r2s.append(val_r2)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}')
        print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val R2: {val_r2:.4f}')

    return train_losses, val_losses, train_rmses, val_rmses, train_r2s, val_r2s


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_preds.append(outputs)
            all_targets.append(targets)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    loss = total_loss / len(data_loader)
    rmse = compute_rmse(all_preds, all_targets)
    r2 = compute_r2(all_preds, all_targets)

    return loss, rmse, r2, all_preds, all_targets


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

# Function to drop the appropriate columns based on variable_2d
def drop_columns_based_on_variable(df, variable_2d):
    if isinstance(variable_2d, list):
        if "Nd_max" in variable_2d:
            var_drop = 'lwp'
            if var_drop in df.columns:
                df.drop(columns=[var_drop], inplace=True)
        elif "lwp" in variable_2d:
            var_drop = 'Nd_max'
            if var_drop in df.columns:
                df.drop(columns=[var_drop], inplace=True)
        else:
            print("considering 2 variables")
    else:
        print("Unknown variable_2d")

    return df


# --------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- MAIN CODE --------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
def main():
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--data-type', type=str, default='ref_rad_total', help='name of the variable in the dataframe one')
    arg('--type-model', type=str, default='NN', help='select the model NN, RF')
    arg('--path_dataframes_pca_scaler', type=str, default="data", help='path where is the dataset save as dataframes and pca, scaler')
    arg('--path_models', type=str, default="output/", help='path of the folder to save the outputs')
    arg('--fold-num', type=int, default=1, help='n k-fold to run')  
    arg('--lr', type=float, default=1e-3)
    arg('--n-epochs', type=int, default=30)
    arg('--batch-size', type=int, default=64)
    arg('--channel_number_list', nargs='+', type=int, help='List of the channels to evaluate (no index)')
    # arg('--all_chan_join', type=str, default="true", help='True false to train all the channel of channel number list in the same time')
    arg('--variable_2d', nargs='+', type=str, default=["Nd_max", "lwp", "topography_c", "FR_LAND"], help='List of the variables 2d to use.')

    args = parser.parse_args()

    type_model = args.type_model
    lr = args.lr
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    variable_2d = args.variable_2d

    channel_number_list = args.channel_number_list
    fold_num = args.fold_num
    data_type = args.data_type
    path_dataframes_pca_scaler = args.path_dataframes_pca_scaler
    path_models = args.path_models

    print(" ========================================== ")
    # print_configuration(args)

    # Print all arguments
    print("Arguments received:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
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
  

    # Apply the function
    df_icon_pca_train = drop_columns_based_on_variable(df_icon_pca_train, variable_2d)
    df_icon_pca_val = drop_columns_based_on_variable(df_icon_pca_val, variable_2d)
    df_icon_pca_test = drop_columns_based_on_variable(df_icon_pca_test, variable_2d)


    
    print(f" Number of features: {df_icon_pca_train.shape[1]}, Number of outputs: {df_ref_train.shape[1]}")


    name_saving_files = f"{type_model}_k_fold_{fold_num}"

    channel_number_list_idx = np.array([x - 1 for x in channel_number_list])
    print(f"--------- indexes channels: {channel_number_list_idx}")
    channel_groups = []
    channel_groups = [(channel_number_list_idx, data_type)] 


    for channels, name in channel_groups:
        print(f"-----------------training model to {name} ---------------------")

        x_train_np = df_icon_pca_train.to_numpy(dtype=float)
        y_train_np = df_ref_train.iloc[:, channels].to_numpy(dtype=float)
        x_val_np = df_icon_pca_val.to_numpy(dtype=float)
        y_val_np = df_ref_val.iloc[:, channels].to_numpy(dtype=float)

        x_test_np = df_icon_pca_test.to_numpy(dtype=float)
        y_test_np = df_ref_test.to_numpy(dtype=float)


        x_train_tensor = torch.tensor(x_train_np, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
        x_val_tensor = torch.tensor(x_val_np, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)


        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MyModel(input_size=x_train_np.shape[1], output_size=y_train_np.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Train the model
        train_losses, val_losses, train_rmses, val_rmses, train_r2s, val_r2s = train_model(model=model, 
                    train_loader=train_loader, 
                    val_loader=val_loader, 
                    n_epochs=n_epochs, 
                    optimizer=optimizer, 
                    criterion=criterion, 
                    device=device)


        # Save the model
        torch.save(model.state_dict(), f'{path_models}/{name}_{name_saving_files}.pth')

       
        # Call the plotting function
        plot_loss(train_losses=train_losses, 
                  val_losses=val_losses, 
                  path_output=path_models, 
                  type_model=f"{name}_{name_saving_files}")



        # ========== Evaluate your model with x_test, y_test ==========
        # Calculate metrics
        
        _, train_rmse, train_r2, outputs_train, targets_train = evaluate_model(model, train_loader, criterion, device)
        # print(f"Training \n RMSE: {val_rmse:.4f} \n R2: {val_r2:.4f}")
        metrics_train = metric_calculation_all_data(outputs=outputs_train,
                                                    targets=targets_train,
                                                    data_name="training")
        
        _, val_rmse, val_r2, outputs_val, targets_val = evaluate_model(model, val_loader, criterion, device)
        metrics_val = metric_calculation_all_data(outputs=outputs_val,
                                                  targets=targets_val,
                                                  data_name="validation")

        _, test_rmse, test_r2, outputs_test, targets_test = evaluate_model(model, test_loader, criterion, device)
        metrics_test = metric_calculation_all_data(outputs=outputs_test,
                                                   targets=targets_test,
                                                   data_name="testing")

        
        train_times = format_times(times_data['train_times'])
        val_times = format_times(times_data['val_times'])
        test_times = format_times(times_data['test_times'])
      
        
    
        # ========== Create DataFrame ==========
        df_metrics = pd.DataFrame({
            f"Train k={fold_num}": metrics_train,
            f"Validation": metrics_val,
            f"Test ({test_times})": metrics_test,
        }).T

        df_metrics = df_metrics.round(4)
        df_metrics.to_csv(f"{path_models}/table_metrics_{name_saving_files}_{name}_with_scaler.csv")
        print(df_metrics)
        
 
        if train_times == val_times:
            print(f"\n Times used in the cross validation k={fold_num}: {train_times}")
        print(f"Testing times used in k={fold_num}: {test_times} \n")

        flatted_size_img = len(lat_lon_data['lat']) * len(lat_lon_data['lon'])
        
        for i, n_time in enumerate(times_data['test_times']):
            x_test_np = df_icon_pca_test.reset_index(drop=True)[i*flatted_size_img:(i+1)*flatted_size_img].to_numpy(dtype=float)
            y_test_np = df_ref_test.iloc[:, channels].reset_index(drop=True)[i*flatted_size_img:(i+1)*flatted_size_img].to_numpy(dtype=float)
            x_test_tensor = torch.tensor(x_test_np, dtype=torch.float32)
            y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)
            test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            _, test_rmse, test_r2, all_preds, all_targets = evaluate_model(model, test_loader, criterion, device)
            metrics_test = metric_calculation_all_data(outputs=all_preds,
                                                       targets=all_targets,
                                                       data_name=f"testing in T{n_time}")

    

                
            
            



if __name__ == '__main__':
    main()
