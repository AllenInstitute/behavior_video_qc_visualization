
import os
import json
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

DATA_PATH = Path("/data")
json_paths = list(DATA_PATH.rglob("**/*metadata.json"))
print(f"found {len(json_paths)} json files with pca + me results outputs")

def load_pkl_and_json(folder_path):
    """
    Load .pkl and .json files from the specified folder.

    Args:
        folder_path (Path): Path to the folder containing the .pkl and .json files.

    Returns:
        tuple: (Loaded .pkl object, .json dictionary)
    """
    pkl_file = folder_path / "pca_model.pkl"
    json_file = folder_path / "pca_generator_metadata.json"

    if not pkl_file.exists() or not json_file.exists():
        raise FileNotFoundError(f"Expected files not found in {folder_path}")

    # Load .pkl file
    ipca = utils.load_pickle_file(pkl_file)

    # Load .json file
    with json_file.open('r', encoding='utf-8') as f:
        pca_dict = json.load(f)

    return ipca, pca_dict

for json_file in json_paths:
    folder_path = Path(json_file).parent
    ipca, pca_dict = load_pkl_and_json(folder_path)
    results_path = pca_dict['top_results_path']

    spatial_masks = pca_dict['spatial_masks']
    fig = plotting.plot_spatial_masks(spatial_masks) 
    utils.save_figure(fig, save_path=results_path, fig_name = 'pca_spatial_masks.png', dpi=300, bbox_inches="tight", transparent=False)

    explained_variance_ratio = ipca.explained_variance_ratio_
    fig = plotting.plot_explained_variance(explained_variance_ratio)
    utils.save_figure(fig, save_path=results_path, fig_name = 'pca_explained_variance.png', dpi=300, bbox_inches="tight", transparent=False)

    pca_motion_energy = pca_dict['pca_motion_energy']
    x = utils.get_x_trace_sec(pca_motion_energy)
    fig = plotting.plot_pca_components_traces(pca_motion_energy, x)
    utils.save_figure(fig, save_path=results_path, fig_name = 'pca_traces.png', dpi=300, bbox_inches="tight", transparent=False)

if __name__ == "__main__": 
    run()
