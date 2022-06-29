from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.results_path = '/home/trdimp-fusion2/pytracking/tracking_results/'
    settings.network_path = '/home/trdimp-fusion2/checkpoints/ltr/dimp/transformer_dimp/'
    settings.result_plot_path = '/home/trdimp-fusion2/pytracking/result_plots/'

    settings.got_packed_results_path = ''
    settings.got_reports_path = ''

    settings.tn_packed_results_path = ''
    settings.tpl_path = ''

    settings.eotb_path = '/data/img_ext'

    return settings

