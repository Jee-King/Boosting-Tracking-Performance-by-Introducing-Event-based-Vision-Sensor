from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = ''
    settings.eotb_path = '/home/iccd/data/img_ext/'
    settings.network_path = '/home/iccd/data/checkpoints/train/stark_s/baseline/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.result_plot_path = '/home/iccd/Documents/Stark-fusion/lib/test/result_plots/'
    settings.results_path = '/home/iccd/Documents/Stark-fusion/lib/test/tracking_results/'    # Where to store tracking results
    settings.segmentation_path = '/home/iccd/Documents/Stark-fusion/lib/test/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''

    return settings

