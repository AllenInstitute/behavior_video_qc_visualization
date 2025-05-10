import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils

def get_x_trace_sec(start_sec, segment_duration, fps = 60):
    """
    Generate time values in seconds based on start time, segment duration, and frames per second.

    Args:
        start_sec (float): Start time in seconds.
        segment_duration (float): Duration of the segment in seconds.
        fps (int, optional): Frames per second. Defaults to 60.

    Returns:
        np.ndarray: Array of time values in seconds.
    """
    x_trace_seconds = np.round(np.arange(start_sec, segment_duration) / fps, 2)
    return x_trace_seconds


def get_data_trace_ind(start_sec, segment_duration, fps = 60):
    """
    Generate frame indices for a segment of the video based on start time, duration, and frames per second.

    Args:
        start_sec (float): Start time in seconds.
        segment_duration (float): Duration of the segment in seconds.
        fps (int, optional): Frames per second. Defaults to 60.

    Returns:
        np.ndarray: Array of frame indices.
    """
    num_frames = int(fps * segment_duration)
    start_frame = int(fps * start_sec)
    stop_frame = start_frame + num_frames
    data_trace_ind = np.arange(start_frame, stop_frame)
    return data_trace_ind


def plot_spatial_masks(spatial_masks, n_to_plot=3, standardize=True) -> plt.Figure:
    """
    Plot spatial masks for the specified number of principal components.

    Args:
        spatial_masks (np.ndarray): Array of spatial masks for each component.
        n_to_plot (int, optional): Number of components to plot. Defaults to 3.
        standardize (bool, optional): Whether to standardize the masks before plotting. Defaults to True.

    Returns:
        plt.Figure: The generated figure containing the spatial mask plots.
    """
    spatial_masks = np.array(spatial_masks)
    if standardize:
        spatial_masks = utils.standardize_masks(spatial_masks)

    fig, axes = plt.subplots(1, n_to_plot, figsize=(3 * (n_to_plot + 1), 3))
    for i, (ax, mask) in enumerate(zip(axes, spatial_masks)):
        vmax = mask.max()
        vmin = mask.min()
        im = ax.imshow(mask, cmap='bwr', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.axis('off')
        ax.set_title(f'PC {i + 1} mask')

    plt.tight_layout()
    plt.show()
    return fig


def plot_explained_variance(explained_variance_ratio) -> plt.Figure:
    """
    Plot the explained variance ratio of PCA components.

    Args:
        explained_variance_ratio (np.ndarray): Array of explained variance ratios.

    Returns:
        plt.Figure: The generated figure with the variance plot.

    Raises:
        ValueError: If explained_variance_ratio is empty.
    """
    if len(explained_variance_ratio) == 0:
        raise ValueError("explained_variance_ratio cannot be empty.")

    fig, ax = plt.subplots(figsize=(4, 3))
    ev = explained_variance_ratio * 100
    ax.plot(range(1, len(ev) + 1), ev, 'o-', linewidth=2, markersize=5)
    ax.set_title('Variance Explained by PC', fontsize=12)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_xlim([0, 30])
    plt.tight_layout()
    plt.show()

    return fig


def plot_pca_components_traces(pca_motion_energy, x_trace_seconds, component_indices=[0, 1, 2], axes=None):
    """
    Plot PCA component traces over time.

    Args:
        pca_motion_energy (np.ndarray): Array of PCA component values.
        x_trace_seconds (np.ndarray): Time values in seconds.
        component_indices (list, optional): Indices of components to plot. Defaults to [0, 1, 2].
        axes (list, optional): Matplotlib axes. If None, new axes are created.

    Returns:
        tuple: (Figure, Axes)
    """
    pca_motion_energy = np.array(pca_motion_energy)
    if pca_motion_energy.shape[1] < max(component_indices) + 1:
        raise ValueError("Insufficient components in pca_motion_energy array.")

    if axes is None:
        fig, axes = plt.subplots(len(component_indices), 1, figsize=(10, 2 * len(component_indices)))

    for i, ax in enumerate(axes):
        trace = utils.remove_outliers_99(pca_motion_energy[:, component_indices[i]], 99)
        trace, x_trace_seconds = utils.check_traces(trace, x_trace_seconds)
        ax.plot(x_trace_seconds, trace)
        ax.set_ylabel(f'PCA {component_indices[i] + 1}', fontsize=14)
        ax.set_title(f'PCA {component_indices[i] + 1}', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)

    axes[-1].set_xlabel('Time (s)', fontsize=16)
    plt.tight_layout()
    return fig, axes



# def plot_edges(meta_dict,title='Edge Detection', ax=None):
#     """
#     Plot an edge-detected image.

#     Args:
#         title (str): Plot title.
#         ax (optional): Matplotlib axis.

#     Returns:
#         tuple: (Figure, Axis)
#     """
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#     ax.imshow(meta_dict['edges'], cmap='gray')
#     ax.set_title(title, fontsize=16)
#     plt.axis('off')
#     plt.show()
#     return fig, ax


# def plot_histogram_with_stats(meta_dict, ax=None):
#     """
#     Plot a histogram of pixel intensities with mean, median, and skewness.

#     Args:
        

#     Returns:
#         tuple: (Axis, Skewness)
#     """
#     pixel_values=meta_dict['pixel_values']
#     ax.hist(pixel_values, bins=256, range=[0, 256], density=True, color='lightblue', alpha=0.9)
#     mean_val = meta_dict['pixel_hist_mean']
#     median_val = meta_dict['pixel_hist_median']
#     std_dev = meta_dict['pixel_hist_std']
#     skewness = meta_dict['skewness']

#     textstr = f'Skew = {skewness:.2f}'
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     _, ymax = ax.get_ylim()
#     ax.text(100, ymax / 2, textstr, fontsize=12, bbox=props)

#     ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
#     ax.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
#     ax.set_xlabel('Pixel Intensity')
#     ax.set_ylabel('Density')
#     ax.legend(loc='upper left')
#     return ax, skewness


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
