

def _standardize_mean_mask(self, mean_mask: np.ndarray) -> np.ndarray:
    """
    Standardizes the mean mask by subtracting its mean and dividing by its standard deviation.

    Args:
        mean_mask (numpy.ndarray): The input 2D mask to be standardized.

    Returns:
        numpy.ndarray: Standardized mean mask.

    Raises:
        ValueError: If standard deviation is zero, leading to division errors.
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

    # Check for NaN values after standardization
    nan_count = np.isnan(mean_mask).sum()
    if nan_count > 0:
        logger.warning(f"Standardized mean mask contains {nan_count} NaN values.")

    return mean_mask


def _plot_spatial_masks(self) -> plt.Figure:
    """
    Plots spatial masks corresponding to principal components.

    Returns:
        matplotlib.figure.Figure: The figure containing the spatial masks.

    Raises:
        AttributeError: If `self.spatial_masks` is missing or None.
    """
    if not hasattr(self, 'spatial_masks') or self.spatial_masks is None:
        raise AttributeError("spatial_masks attribute is missing or None. Run PCA before plotting.")

    if not isinstance(self.spatial_masks, np.ndarray):
        raise TypeError("spatial_masks must be a NumPy array.")

    n_components = self.n_to_plot
    fig,axes = plt.subplots(figsize=(3 * (n_components + 1), 3), nrows = 1, ncols = 4)  # Extra space for the example frame

    # Plot the example frame
    ax = fig.add_subplot(1, n_components + 1, 1)
    im = ax.imshow(self.example_frame, cmap='gray', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    ax.set_title(f'Example Frame', fontsize=10)

    for i, (ax, mask) in enumerate(zip(axes[1:], self.spatial_masks)):
        im = ax.imshow(mask, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
        ax.set_title(f'PC {i + 1} mask')

    plt.tight_layout()
    plt.show()

    return fig


def _plot_explained_variance(self) -> plt.Figure:
    """
    Plots the explained variance ratio of each principal component from a PCA model.

    Returns:
        matplotlib.figure.Figure: The figure containing the explained variance plot.

    Raises:
        ValueError: If PCA model has not been fitted.
    """
    if not hasattr(self.pca, 'explained_variance_ratio_'):
        raise ValueError("PCA object must be fitted before plotting.")

    fig, ax = plt.subplots(figsize=(4, 3))
    explained_variance = self.pca.explained_variance_ratio_ * 100

    ax.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2, markersize=5)
    ax.set_title('Variance Explained by Principal Components', fontsize=12)
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Explained Variance (%)', fontsize=12)
    ax.set_xlim([0, 30])  # Show only top 30 components
    plt.tight_layout()
    plt.show()

    return fig


def _plot_pca_components_traces(self, component_indices: list = [0, 1, 2], remove_outliers: bool = False, axes=None) -> plt.Figure:
    """
    Plots PCA components against time.

    Args:
        component_indices (list, optional): Indices of PCA components to plot. Defaults to [0, 1, 2].
        remove_outliers (bool, optional): Whether to remove outliers above 99%. Defaults to True.
        axes (matplotlib.axes.Axes, optional): Axes for plotting. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The figure containing PCA component traces.

    Raises:
        ValueError: If PCA motion energy data is missing or has fewer than three components.
        AssertionError: If timestamps and PCA traces have different lengths.
    """
    if not hasattr(self, 'pca_motion_energy') or self.pca_motion_energy is None:
        raise ValueError("PCA motion energy data is missing. Run PCA before plotting.")

    if self.pca_motion_energy.shape[1] < 3:
        raise ValueError("pca_motion_energy must have at least 3 components to plot.")

    fps = self.video_metadata.get('fps', 60)  # Default FPS to 60 if missing
    x_range = min(10000, self.pca_motion_energy.shape[0])  # Ensure range doesn't exceed available data
    x_trace_seconds = np.round(np.arange(0, x_range) / fps, 2)

    if axes is None:
        fig, axes = plt.subplots(len(component_indices), 1, figsize=(15, 2 * len(component_indices)))

    for i, ax in enumerate(axes):
        pc_trace = self.pca_motion_energy[:x_range, component_indices[i]]
        if remove_outliers:
            pc_trace = utils.remove_outliers_99(pc_trace)

        assert len(x_trace_seconds) == len(pc_trace), "Timestamps and PC trace lengths do not match."

        ax.plot(x_trace_seconds, pc_trace)
        ax.set_ylabel(f'PCA {component_indices[i] + 1}', fontsize=16)
        ax.set_title(f'PCA {component_indices[i] + 1} over time (s)', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)

    axes[-1].set_xlabel('Time (s)', fontsize=16)
    plt.tight_layout()
    return fig

def _get_motion_energy_trace(self):
    npz_data = np.load(self.npz_path)

    if not npz_data:
        raise ValueError("No data found in the NPZ file.")
    
    if self.use_cropped_frames:
        array = npz_data['cropped_frame_motion_energy']
    else:
        array = npz_data['full_frame_motion_energy']

    self.motion_energy_trace = array
    return self
    

def _plot_motion_energy_trace(self, remove_outliers: bool = True) -> plt.Figure:
    """
    Creates a figure and plots a NumPy array from an NPZ file.

    Args:

    Returns:
        plt.Figure: The matplotlib figure object.

    Raises:
        ValueError: If the specified array name is not found.
    """

    if remove_outliers:
        array = utils.remove_outliers_99(self.motion_energy_trace)
    else:
        array = self.motion_energy_trace

    fig, ax = plt.subplots(figsize=(15, 6))
    if self.use_cropped_frames or self.recrop:
        array_name = 'cropped frames'
    else:
        array_name = 'full frames'


    ax.plot(array, label=f"{array_name}")
    ax.set_title(f"Motion energy trace for {array_name} from NPZ")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

    logger.info(f"Plotted array: {array_name}")

    return fig  

