import torch.nn as nn

from .generationBased import *
from .subNets import AlignSubNet

__all__ = ['ConstructModels']

class ConstructModels(nn.Module):
    def __init__(self, args):
        super(ConstructModels, self).__init__()
        self.MODEL_MAP = {
            'cyin': CyIN,
            
        }

        self.need_model_aligned = args['need_model_aligned']
        # Simulating word-align network (text_length == audio_length == vision_length)
        if (self.need_model_aligned):
            self.alignNet = AlignSubNet(args, mode='avg_pool') # mode in ['avg_pool', 'ctc', 'conv1d']
            if 'seq_lens' in args.keys():
                args['seq_lens'] = self.alignNet.get_seq_len()

        lastModel = self.MODEL_MAP[args['model_name']]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, vision_x, *args, **kwargs):
        if (self.need_model_aligned):
            text_x, audio_x, vision_x = self.alignNet(text_x, audio_x, vision_x)
        return self.Model(text_x, audio_x, vision_x, *args, **kwargs)
