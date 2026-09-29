from .generationBased import *

__all__ = ['ConstructTrains']

class ConstructTrains():
    def __init__(self):
        self.TRAIN_MAP = {
            'cyin': CyIN,
            
        }

    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args)
