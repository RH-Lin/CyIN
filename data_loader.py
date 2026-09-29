import logging
import json
import pickle

import torch
from torch.utils.data import DataLoader, Dataset, distributed
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

import lmdb
from utils import LoadAudioVision

__all__ = ['MMDataLoader']

logger = logging.getLogger('CyIN')

class MMDataset(Dataset): # MMDataset is Dataset's subclass; Dataset is MMDataset's superclass
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        if self.args.train_mode == "regression":
            DATASET_MAP = {
                'mosi': self.__init_mosi,
                'mosei': self.__init_mosei,
            }
        elif self.args.train_mode == "recognition":
            DATASET_MAP = {
                'mosei': self.__init_mosei_emo,
                'iemocap4': self.__init_iemocap_emo,
                'iemocap6': self.__init_iemocap_emo,
                'iemocap9': self.__init_iemocap_emo,
                'meld': self.__init_meld_emo,
            }
        DATASET_MAP[args['dataset_name']]()

    def __len__(self):
        return len(self.labels['M']) # Return the size of self-defined dataset for Dataset's latter ergodic

    def __getitem__(self, index): # Support index access and preprocess corresponding data
        # read audio and video from lmdb by video_ids
        # audio, vision = LoadAudioVision.load_audiovision_lmdb(self.txn, self.ids[index])

        sample = {
            'text': self.text[index],
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
        }
        return sample

    def __init_mosi(self):
        # read audio and video extracted features from pkl
        data_dict = pickle.load(open(self.args['feature_path'], 'rb'))

        self.text = np.array(data_dict[self.mode]['text']) #raw text
        self.audio = data_dict[self.mode]['audio']
        self.vision = data_dict[self.mode]['vision']
        self.ids = data_dict[self.mode]['id']
        self.labels = {
            'M': np.array(data_dict[self.mode]['regression_labels']['M']).astype(np.float32)
        }

        logger.info(f"{self.mode.upper()} samples: {len(self.labels['M'])}")

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_mosei_emo(self):
        data_dict = pickle.load(open(self.args['feature_path'], 'rb'))
        if self.args.transformers == 'deberta':
            # replace </s> with [SEP]
           data_dict[self.mode]['text'] = [s.replace("</s>", "[SEP]") for s in data_dict[self.mode]['text']]

        self.text = np.array(data_dict[self.mode]['text']) #raw text
        self.audio = data_dict[self.mode]['audio']
        self.vision = data_dict[self.mode]['vision']
        self.ids = data_dict[self.mode]['id']

        self.labels = {
            'M': np.array(data_dict[self.mode]['recognition_labels']['M']).astype(np.int32)
        }

        logger.info(f"{self.mode.upper()} samples: {len(self.labels['M'])}")

    def __init_iemocap_emo(self):
        return self.__init_mosei_emo()

    def __init_meld_emo(self):
        return self.__init_mosei_emo()

    def __init_urfunny_emo(self):
        data_dict = pickle.load(open(self.args['feature_path'], 'rb'))

        self.text = np.array(data_dict[self.mode]['text'])  # raw text
        self.audio = data_dict[self.mode]['audio']
        self.vision = data_dict[self.mode]['vision']
        self.ids = data_dict[self.mode]['id']

        self.labels = {
            'M': np.array(data_dict[self.mode]['classification_labels']['M']).astype(np.int32)
        }

        logger.info(f"{self.mode.upper()} samples: {len(self.labels['M'])}")

    def __init_mustard_emo(self):
        return self.__init_urfunny_emo()

    def get_pos_weight(self, mu=1.0):
        pos_nums = torch.sum(torch.tensor(self.labels['M']), dim=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def get_class_weight(self, label):
        if self.args['dataset_name'] == 'mosei':
            return torch.tensor(
                [compute_class_weight('balanced',classes=np.unique(label[:,i]), y=label[:,i]
            )[1] for i in range(label.shape[-1])], dtype=torch.float
            )
        else:
            label_cls = np.argmax(label, axis=1)
            return torch.tensor(
                compute_class_weight('balanced',classes=np.unique(label_cls), y=label_cls), dtype=torch.float
            )

def MMDataLoader(args, num_workers):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    } # To pass created Dataset subclass as parameters to builded DataLoader class

    if args.train_mode == "recognition":
        args['pos_weight'] = datasets['train'].get_pos_weight()
        args['class_weight'] = datasets['train'].get_class_weight(label=datasets['train'].labels['M']) # 1 /

    if 'sample_batch_size' not in args:
        args['sample_batch_size'] = args['batch_size']

    if args.distributed == True:
        sampler = {
            'train': distributed.DistributedSampler(datasets['train'], shuffle=True),
            'valid': None,
            'test': None
        } # not use DDP for valid/test

        dataLoader = {
            ds: DataLoader(datasets[ds],
                           batch_size=args['batch_size'] if ds == 'train' else args['sample_batch_size'],
                           num_workers=num_workers,
                           sampler=sampler[ds],
                           drop_last=False)
            for ds in datasets.keys()
        }
        dataLoader['train_init'] = DataLoader(datasets['train'],
                           batch_size=args['batch_size'],
                           num_workers=num_workers,
                           sampler=sampler['train'],
                           drop_last=False)
    else:
        dataLoader = {
            ds: DataLoader(datasets[ds],
                           batch_size=args['batch_size'] if ds == 'train' else args['sample_batch_size'],
                           num_workers=num_workers,
                           shuffle=True,
                           drop_last=False)
            for ds in datasets.keys()
        }
        dataLoader['train_init'] = DataLoader(datasets['train'],
                           batch_size=args['batch_size'],
                           num_workers=num_workers,
                           drop_last=False)

    return dataLoader


