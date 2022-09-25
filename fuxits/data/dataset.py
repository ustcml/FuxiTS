from builtins import isinstance
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from typing import Tuple, Union, Dict, List
import importlib
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
import typing
from fuxits.utils.timeparse import stepparse, timeparse

def collate(batch):
    x, y = default_collate(batch)
    return [b.transpose(1, -1) for b in x], y  # B x T x N x F -> B x F x N x T

class TSDataset(Dataset):

    r"""
    state is set by class construction function.
    """

    def __init__(self, name:str='METR_LA') -> None:
        data_class = getattr(importlib.import_module('fuxits.data.traffic.preprocess'), name.upper())
        data = data_class().build()
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                scaler = StandardScaler()
                v = scaler.fit_transform(v.reshape(v.shape[0], -1)).reshape(v.shape)
                setattr(self, k, torch.from_numpy(v).type(torch.float32))
            else:
                setattr(self, k, v)

    @property
    def num_steps(self):
           return [self.in_steps] + [step for step, _ in self.extra_input]
    
    @property
    def datafeatures(self):
        result = self.features
        result.update({'hist_steps': self.num_steps, 'pred_steps': self.out_steps})
        return result

    def num_points(self, units='hours'):
        if units == 'hours':
            return 60 // self.sample_freq
        elif units == 'days':
            return 24 * 60 // self.sample_freq
        elif units == 'weeks':
            return 7 * 24 * 60 // self.sample_freq
        else:
            raise ValueError(f'{units} not supported')

    def __getitem__(self, index:int):
        anchor = self.anchors[index]
        if isinstance(self.out_steps, int):
            y = self.state[anchor : anchor+self.out_steps]
        else:
            y = self.state[[anchor + _ - 1 for _ in self.out_steps]]
        x = self.state[anchor-self.in_steps : anchor]
        x_final = [x]
        for (steps, num_points) in self.extra_input:
            extra_idx = np.concatenate([np.arange(idx, idx+self.out_steps) \
                                for idx in np.arange(anchor-steps*num_points, anchor, num_points)])
            x_final = x_final + [self.state[extra_idx]]
        return x_final, y
    
    def __len__(self):
        return len(self.anchors)

    def loader(self, batch_size, num_workers=3, shuffle=False, drop_last=False):
        output = DataLoader(self, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last, collate_fn=collate)
        return output

    def build(self, \
            ratio:Union[Tuple[int,int,int], Tuple[int,int]], 
            in_steps:Union[int, List[str]],
            out_steps:Union[int, str, List[int]]):
        r"""Build a dataset and split them into train, validation and test dataset.
        for efficiency, we only keep index into these three subsets.
        Args:
            ratio: split ratio, specified for train-test, or train-vali-test
            in_steps:  an integer indicates how many steps are used as features
                       a list of str imply how many steps as well as the inteveral between steps
                       e.g. [12, 3h, 2d, 6w] means preceding 12 steps, 2 steps with one day interval as well as 6 steps with one week inteveral
            out_steps: how many steps to predict (int), and which steps to predict (List[int])
        Returns:
            list: A list contains train/vali/test data- [train, vali, test]
        """
        if isinstance(in_steps, list):
            if len(in_steps) > 1:
                assert all(not s.isdigit() for s in in_steps[1:])
                extra_input = stepparse(in_steps[1:])
                extra_input = [(int(v), self.num_points(k)) for k, v in extra_input.items()]
            else:
                extra_input = []
                #extra_input is a tuple of two elements
                #first parameter: how many steps in preceding to prediction
                #second parameter: how many data points between steps
            if isinstance(in_steps[0], int):
                in_steps = in_steps[0]
            elif in_steps[0].isdigit():
                in_steps = int(in_steps[0])
            else:
                assert in_steps[0].endswith('h')
                in_steps = timeparse(in_steps[0]) // 60 // self.sample_freq
        
        if isinstance(out_steps, str):
            out_steps = timeparse(out_steps) // 60 // self.sample_freq
        if isinstance(out_steps, int):
            assert in_steps % out_steps == 0
        else:
            out_steps = np.array(out_steps)

        T = self.state.shape[0]
        splits = np.multiply(T, ratio).astype(np.int32)
        splits[0] = T - splits[1:].sum()
        if len(extra_input) > 0:
            valid_start = max(max(steps * num_points for steps, num_points in extra_input), in_steps)
        else:
            valid_start = in_steps
        splits = np.hstack([[valid_start], np.cumsum(splits)])
        data_idx = [(splits[i-1], splits[i] - out_steps if isinstance(out_steps, int) else out_steps.max()-1) 
                    for i in range(1, splits.shape[0])]
        anchors = [np.arange(*e) for e in data_idx]
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.extra_input = extra_input
        return [self._copy(_) for _ in anchors]
    
    def _copy(self, anchors):
        d = copy.copy(self)
        d.anchors = anchors
        return d

if __name__=='__main__':
    dataset = TSDataset()
    train, test = dataset.build([0.8,0.2], [dataset.num_points('hours'), '1d', '2w'], dataset.num_points('hours'))
    #train, test = dataset.build([0.8,0.2], 2*dataset.num_points('hours'), dataset.num_points('hours'),
    #                                (1, dataset.num_points('days')), 
    #                                2, dataset.num_points('weeks'))

    for x, y in train.loader(10):
        print(len(x))
        print(y.shape)
        break
    #print(test[1])
    #print(len(train))
