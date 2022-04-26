from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters(netepoch=None):
    params = TrackerParams()
    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    params.net = NetWithBackbone(net_path=netepoch,
                                 use_gpu=params.use_gpu)
    return params
