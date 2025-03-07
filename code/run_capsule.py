
#plot and save fig
fig = me_pca._plot_spatial_masks()
utils.save_figure(fig, save_path=me_pca.top_results_path, fig_name = 'pca_spatial_masks.png', dpi=300, bbox_inches="tight", transparent=False)

fig = me_pca._plot_explained_variance()
utils.save_figure(fig, save_path=me_pca.top_results_path, fig_name = 'pca_explained_variance.png', dpi=300, bbox_inches="tight", transparent=False)

fig = me_pca._plot_motion_energy_trace()
utils.save_figure(fig, save_path=me_pca.top_results_path, fig_name = 'motion_energy_trace.png', dpi=300, bbox_inches="tight", transparent=False)

try:
    fig = me_pca._plot_pca_components_traces()
    utils.save_figure(fig, save_path=me_pca.top_results_path, fig_name = 'pca_components_traces.png', dpi=300, bbox_inches="tight", transparent=False)
except:
    print('couldnt plot pca traces')