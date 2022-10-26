from turtle import forward
from fuxits.predictor.predictor import Predictor
import torch.nn as nn
class DCRNN(Predictor):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, x):
        pass



