import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Union, Dict, List
import importlib
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler

class TSDataset(Dataset):

    r"""
    state is set by class construction function.
    """

    def __init__(self, name:str='METR_LA') -> None:
        data_class = getattr(importlib.import_module('data.statepreproc'), name.upper())
        data = data_class().build()
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                scaler = StandardScaler()
                v = scaler.fit_transform(v)
                setattr(self, k, torch.from_numpy(v))
            else:
                setattr(self, k, v)


    def num_points(self, units='hour'):
        if units == 'hour':
            return 60 // self.sample_freq
        elif units == 'day':
            return 24 * 60 // self.sample_freq
        elif units == 'week':
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
        outputs = {'x':x, 'y':y}
        for iter, (steps, num_points) in enumerate(self.extra_input):
            extra_idx = np.concatenate([np.arange(idx, idx+self.out_steps) \
                                for idx in np.arange(anchor-steps*num_points, anchor, num_points)])
            outputs['x'+str(iter)] = self.state[extra_idx]
        return outputs
    
    def __len__(self):
        return len(self.anchors)

    def loader(self, batch_size, num_works=3, shuffle=False, drop_last=False):
        output = DataLoader(self, batch_size=batch_size, num_workers=num_works, shuffle=shuffle, drop_last=drop_last)
        return output

    def build(self, \
            ratio:Union[Tuple[int,int,int], Tuple[int,int]], 
            in_steps:int, 
            out_steps:Union[int, List[int]],
            *extra_input:Tuple[int, int]):
        r"""Build a dataset and split them into train, validation and test dataset.
        for efficiency, we only keep index into these three subsets.
        Args:
            ratio: split ratio, specified for train-test, or train-vali-test
            in_steps: how many steps are used as features
            out_steps: how many steps to predict (int), and which steps to predict (List[int])
            extra_input: other features to use
                first parameter: how many steps in preceding to prediction
                second parameter: how many data points between steps
        Returns:
            list: A list contains train/vali/test data- [train, vali, test]
        """
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


dataset = TSDataset()
train, test = dataset.build([0.8,0.2], 2*dataset.num_points('hour'), dataset.num_points('hour'),
                                (1, dataset.num_points('day')), 
                                2, dataset.num_points('week'))

for data in train.loader(10):
    for k, v in data.items():
        print(k, v.shape)
    break
#print(test[1])
#print(len(train))
