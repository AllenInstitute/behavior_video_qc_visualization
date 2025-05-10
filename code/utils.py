

import numpy as np 
import os
import pickle 
import matplotlib.pyplot as plt


def load_pickle_file(file_path: str):
    """
    Load a pickle file and return the deserialized object.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        object: The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is corrupted or not a valid pickle file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at '{file_path}' does not exist.")

    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Error loading pickle file: {e}")


def load_npz_file(npz_file_path: str) -> dict:
    """
    Loads a NumPy `.npz` file and returns its contents.

    Args:
        npz_file_path (str): Path to the `.npz` file.

    Returns:
        dict: Dictionary containing NumPy arrays from the NPZ file.

    Raises:
        FileNotFoundError: If the specified NPZ file does not exist.
        ValueError: If the NPZ file is empty.
    """
    
    npz_data = np.load(npz_file_path)
    return npz_data


def get_x_trace_sec(me_frames: np.ndarray, fps: int = 60) -> np.ndarray:
    """
    Generate an x-axis trace in seconds for motion energy frames.

    Args:
        me_frames (numpy.ndarray): Array of motion energy frames.
        fps (int, optional): Frames per second. Defaults to 60.

    Returns:
        numpy.ndarray: Time values in seconds.
    """
    return np.round(np.arange(1, me_frames.shape[0]+1) / fps, 2)


def remove_outliers_99(arr: np.ndarray, percentile = 99) -> np.ndarray:
    """
    Removes outliers above the 99th percentile in a NumPy array.

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Array with outliers removed.
    """
    threshold = np.percentile(arr, percentile)
    arr_out = arr.copy()  # Avoid modifying the original array
    arr_out[arr_out > threshold] = np.nan
    return arr_out


def save_figure(fig: plt.Figure, save_path: str, fig_name: str, dpi: int = 300,
                bbox_inches: str = "tight", transparent: bool = True) -> None:
    """
    Save a Matplotlib figure to a specified path.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        save_path (str): Directory where the figure should be saved.
        fig_name (str): Filename of the saved figure.
        dpi (int, optional): Resolution in dots per inch. Defaults to 300.
        bbox_inches (str, optional): Trim white space. Defaults to "tight".
        transparent (bool, optional): Save with transparent background. Defaults to True.
    """
    figpath = os.path.join(save_path, fig_name)
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    fig.savefig(figpath, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
    print(f"Figure saved at: {figpath}")


def check_traces(x, y):
    """
    Ensures x and y are not empty and have the same length.
    If one is longer, trims it to match the length of the shorter.

    Args:
        x (array-like): First sequence.
        y (array-like): Second sequence.

    Returns:
        tuple: Trimmed x and y arrays of equal length.

    Raises:
        ValueError: If x or y is empty.
    """
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        raise ValueError("Input arrays must not be empty.")

    if len(x) == len(y):
        return x, y

    min_len = min(len(x), len(y))
    return x[:min_len], y[:min_len]
    

def load_pickle_file(file_path: str):
    """
    Load a pickle file and return the deserialized object.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        object: The deserialized Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is corrupted or not a valid pickle file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at '{file_path}' does not exist.")

    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f"Error loading pickle file: {e}")


def standardize_masks(masks) -> np.ndarray:
    """
    Standardizes a 2D mask by subtracting its mean and dividing by its standard deviation.

    Args:
        mean_mask (np.ndarray): The input 2D array.

    Returns:
        np.ndarray: The standardized mask.

    Raises:
        ValueError: If the standard deviation is zero.
        TypeError: If input is not a NumPy array.
    """
    if not isinstance(masks, np.ndarray):
        raise TypeError("mean_mask must be a NumPy array.")

    mean = masks.mean(axis=1).mean(axis=1)
    std = masks.mean(axis=1).mean(axis=1)
    mean = np.reshape(mean, (mean.shape[0],1,1))
    std = np.reshape(std, (std.shape[0],1,1))

    # Validate standard deviation
    if std.any() == 0:
        raise ValueError("Standard deviation is zero, leading to division errors!")

    # Standardize the mean mask
    standardized_masks = (masks - mean) / std

    if np.isnan(standardized_masks).any():
        logger.warning("Standardized mean mask contains NaN values.")

    return standardized_masks