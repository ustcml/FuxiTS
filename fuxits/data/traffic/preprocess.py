import os
import pandas as pd
import numpy as np
from fuxits.utils.timeparse import timeparse
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

config = parser_yaml("fuxits/data/dataset.yml")


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

    def collect_rawdata(self, flowname='metr-la.h5', adjname=['distances_la_2012.csv', 'metr_ids.txt']):
        flow = pd.read_hdf(os.path.join(self.data_meta['url'], flowname)).values
        sensor_ids_file = os.path.join(self.data_meta['url'], adjname[1])
        distance_file = os.path.join(self.data_meta['url'], adjname[0])
        with open(sensor_ids_file) as f:
            sensor_ids = dict((id, i) for i, id in enumerate(f.read().strip().split(',')))
        dist_df = pd.read_csv(distance_file,dtype={'from': 'str', 'to': 'str'})
        dist_df = dist_df[dist_df['from'].isin(sensor_ids) & dist_df['to'].isin(sensor_ids)]
        dist_df.replace({'from':sensor_ids, 'to':sensor_ids}, inplace=True)
        return np.expand_dims(flow, -1), (dist_df['from'], dist_df['to'], dist_df['cost']), len(sensor_ids)
    
    def convert_rawdata(self, flow, coo, num_nodes):
        def convert_adj():
            A = np.zeros([num_nodes]*2)
            A[coo[0], coo[1]] = coo[2]
            return A
        return {'state':flow, 
                'sample_freq': timeparse(self.data_meta['sample_freq'])//60,
                'features': {'num_nodes': flow.shape[1], 
                            'in_channels': flow.shape[-1],
                            'static_adj':convert_adj()}
                }
        

class LOS_LOOP(TrafficStatePreproc):
    pass

class SZ_TAXI(TrafficStatePreproc):
    pass

class LOOP_SEATTLE(TrafficStatePreproc):
    pass

class Q_TRAFFIC(TrafficStatePreproc):
    pass