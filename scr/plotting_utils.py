
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import numpy as np



def plot_loss_train_val(history, type_model, path_output):
    """
    Plots training and validation loss from the training history and saves the plot to a specified path.

    This function visualizes the loss for training and validation data over epochs using matplotlib.
    It then saves the resulting plot as a PNG file in the specified output directory.

    Parameters:
        history (History): History object from the training of a model. Expected to have 'loss' and 'val_loss' keys.
        type_model (str): Type or identifier of the model being used. Used in naming the saved plot.
        path_output (str): Directory path where the plot should be saved.

    Returns:
        None. A plot is saved to the specified directory.

    Notes:
        The saved plot is named in the format: 'history_{type_model}.png'.
    """
    f = plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    figure_name = f'{path_output}/history_{type_model}.png'
    f.tight_layout()
    f.savefig(figure_name, dpi=60)


