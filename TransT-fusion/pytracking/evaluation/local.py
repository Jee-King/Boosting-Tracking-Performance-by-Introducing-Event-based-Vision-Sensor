from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.eotb_path = '/home/iccd/data/img_ext/'
    settings.network_path = '/home/iccd/Documents/TransT-fusion/checkpoints/ltr/transt/transt'  # Where tracking networks are stored.
    settings.result_plot_path = '/home/iccd/Documents/TransT-fusion/pytracking/result_plots/'
    settings.results_path = '/home/iccd/Documents/TransT-fusion/pytracking/tracking_results'  # Where to store tracking results
    settings.segmentation_path = '/home/iccd/Documents/TransT-fusion/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''

    return settings

