import imp
import os
import pandas as pd
import numpy as np
from utils.timeparse import timeparse
def parser_yaml(config_path):
    import yaml,re
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(
            u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X
        ), list(u'-+0123456789.')
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        ret = yaml.load(f.read(), Loader=loader)
    return ret

config = parser_yaml("data/dataset.yml")


class TrafficStatePreproc:

    def collect_rawdata(self, **kwargs):
        r""" collect rawdata from dataset url
        Returns:
            a list of data structure
        """
        pass

    
    def convert_rawdata(self, *args):
        r""" convert rawdata into standard format.
        Returns:
            a traffic state tensor of size TxNxC, or TxMxNxC, or TxNxNxC or TxNxMxNxMxC
        """
        pass
    
    def build(self):
        data_list = self.collect_rawdata()
        return self.convert_rawdata(*data_list)

class METR_LA(TrafficStatePreproc):
    

    def __init__(self) -> None:
        self.data_meta = config[__class__.__name__]

    def collect_rawdata(self, flowname='metr-la.h5', adjname='W_metrla.csv'):
        flow = pd.read_hdf(os.path.join(self.data_meta['url'], flowname)).values
        A = pd.read_csv(os.path.join(self.data_meta['url'], adjname)).values
        return flow, A
    
    def convert_rawdata(self, flow, A):
        def convert_adj(W:np.ndarray, sigma2=0.1, epsilon=0.5):
            n = W.shape[0]
            W = W /10000
            W[W<np.finfo(float).eps] = np.inf
            output = np.exp(-W * W / sigma2) 
            output[output<epsilon] = 0.
            np.fill_diagonal(output, 0.)
            return output
        return {'state':flow, 
                'adjacency':convert_adj(A), 
                'sample_freq': timeparse(self.data_meta['sample_freq'])//60 }
        

class LOS_LOOP(TrafficStatePreproc):
    pass

class SZ_TAXI(TrafficStatePreproc):
    pass

class LOOP_SEATTLE(TrafficStatePreproc):
    pass

class Q_TRAFFIC(TrafficStatePreproc):
    pass