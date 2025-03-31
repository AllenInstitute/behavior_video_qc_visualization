import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_x_trace_sec(start_sec, segment_duration, fps = 60):
    x_trace_seconds = np.round(np.arange(start_sec, segment_duration) / fps, 2)
    return x_trace_seconds

def get_data_trace_ind(start_sec, segment_duration, fps = 60):
    num_frames = int(fps * segment_duration)
    start_frame = int(fps * start_sec)
    stop_frame = start_frame+num_frames
    data_trace_ind = np.arange(start_frame, stop_frame)

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

def standardize_mean_mask(self, mean_mask: np.ndarray) -> np.ndarray:
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
    if not isinstance(mean_mask, np.ndarray):
        raise TypeError("mean_mask must be a NumPy array.")

    mean = mean_mask.mean()
    std = mean_mask.std()

    # Validate standard deviation
    if std == 0:
        raise ValueError("Standard deviation is zero, leading to division errors!")

    # Standardize the mean mask
    mean_mask = (mean_mask - mean) / std

    if np.isnan(mean_mask).any():
        logger.warning("Standardized mean mask contains NaN values.")

    return mean_mask


def plot_spatial_masks(self) -> plt.Figure:
    """
    Plot spatial masks for principal components and an example frame.

    Returns:
        plt.Figure: The matplotlib figure object.

    Raises:
        AttributeError: If spatial_masks is not available.
    """
    if not hasattr(self, 'spatial_masks') or self.spatial_masks is None:
        raise AttributeError("Missing spatial_masks. Run PCA before plotting.")

    n_components = self.n_to_plot
    fig, axes = plt.subplots(1, n_components + 1, figsize=(3 * (n_components + 1), 3))

    ax = axes[0]
    im = ax.imshow(self.example_frame, cmap='gray', aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.axis('off')
    ax.set_title('Example Frame', fontsize=10)

    for i, (ax, mask) in enumerate(zip(axes[1:], self.spatial_masks)):
        im = ax.imshow(mask, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        ax.set_title(f'PC {i + 1} mask')

    plt.tight_layout()
    plt.show()

    return fig


def plot_explained_variance(self) -> plt.Figure:
    """
    Plot the explained variance ratio from a fitted PCA model.

    Returns:
        plt.Figure: The matplotlib figure object.

    Raises:
        ValueError: If PCA model is not fitted.
    """
    if not hasattr(self.pca, 'explained_variance_ratio_'):
        raise ValueError("PCA model is not fitted.")

    fig, ax = plt.subplots(figsize=(4, 3))
    ev = self.pca.explained_variance_ratio_ * 100

    ax.plot(range(1, len(ev) + 1), ev, 'o-', linewidth=2, markersize=5)
    ax.set_title('Variance Explained by PC', fontsize=12)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_xlim([0, 30])
    plt.tight_layout()
    plt.show()

    return fig


def plot_pca_components_traces(pca_dict, x_trace_seconds, component_indices=[0, 1, 2], axes=None):
    """
    Plot selected PCA components against time.

    Args:
        x_trace_seconds (np.ndarray): Time values in seconds.
        component_indices (list): PCA component indices to plot.
        axes (optional): Matplotlib axes.

    Returns:
        tuple: (Figure, Axes)
    """
    pca_motion_energy = pca_dict['pca_motion_energy_traces']
    if pca_motion_energy.shape[1] < 3:
        raise ValueError("pca_motion_energy must have at least 3 components.")

    if axes is None:
        fig, axes = plt.subplots(len(component_indices), 1, figsize=(10, 2 * len(component_indices)))

    for i, ax in enumerate(axes):
        ax.plot(x_trace_seconds, pca_motion_energy[:, component_indices[i]])
        ax.set_ylabel(f'PCA {component_indices[i] + 1}', fontsize=16)
        ax.set_title(f'PCA {component_indices[i] + 1} over time (s)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)

    axes[-1].set_xlabel('Time (s)', fontsize=16)
    return fig, axes


def plot_edges(meta_dict,title='Edge Detection', ax=None):
    """
    Plot an edge-detected image.

    Args:
        title (str): Plot title.
        ax (optional): Matplotlib axis.

    Returns:
        tuple: (Figure, Axis)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(meta_dict['edges'], cmap='gray')
    ax.set_title(title, fontsize=16)
    plt.axis('off')
    plt.show()
    return fig, ax


def plot_histogram_with_stats(meta_dict, ax=None):
    """
    Plot a histogram of pixel intensities with mean, median, and skewness.

    Args:
        

    Returns:
        tuple: (Axis, Skewness)
    """
    pixel_values=meta_dict['pixel_values']
    ax.hist(pixel_values, bins=256, range=[0, 256], density=True, color='lightblue', alpha=0.9)
    mean_val = meta_dict['pixel_hist_mean']
    median_val = meta_dict['pixel_hist_median']
    std_dev = meta_dict['pixel_hist_std']
    skewness = meta_dict['skewness']

    textstr = f'Skew = {skewness:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    _, ymax = ax.get_ylim()
    ax.text(100, ymax / 2, textstr, fontsize=12, bbox=props)

    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Density')
    ax.legend(loc='upper left')
    return ax, skewness


def add_crop(frame, crop_region=None):
    """
    add a visual crop to the frame

    Args:
        frame (numpy.ndarray): The input frame.
        crop_region (tuple, optional): Coordinates of the region to crop (x, y, width, height).

    Returns:
        numpy.ndarray: frame with the cropped region highlighted by a red rectangle, if crop_region is not None.
    """
    if crop_region is not None:
        x, y, w, h = crop_region
        frame_with_crop = cv2.rectangle(frame.copy(), (x, y), (x+w, y+h), (0, 0, 255), 2)
        return frame_with_crop
    else:
        print("no crop region was provided")
        return frame


def plot_frame_with_crop(frame, crop_region, axes):
    """
    Plot a frame with a highlighted crop region.

    Args:
        frame (np.ndarray): Original frame.
        crop_region (tuple): (x, y, w, h) crop region.
        axes (list): Two matplotlib axes.

    Returns:
        list: Axes with plots.
    """
    frame_with_crop = add_crop(frame, crop_region)
    frame_with_crop_rgb = cv2.cvtColor(frame_with_crop, cv2.COLOR_BGR2RGB)
    cropped_frame = crop_frame(frame, crop_region)

    axes[0].imshow(frame_with_crop_rgb)
    axes[0].set_title('Frame with Cropped Region')
    axes[1].imshow(cropped_frame)
    axes[1].set_title('Cropped Region')
    return axes
