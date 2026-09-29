import json
from pathlib import Path
import random
import os
from easydict import EasyDict as edict

def get_config_train(model_name, dataset_name, config_file=''):
    """
    Get the regression config of given dataset and model from config file.
    Parameters:
        model_name (str): Name of model.
        dataset_name (str): Name of dataset.
        config_file (str): Path to config file, if given an empty string, will use default config file.
    Returns:
        config (dict): config of the given dataset and model
    """
    if config_file == '':
        config_file = Path(__file__).parent / "config" / "config_regression.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name]
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (
                model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(dataset_args)
    config['video_path'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['video_path'])
    config['annotate_path'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['annotate_path'])
    config['feature_path'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['feature_path'])

    config = edict(config)  # Edict allows access to dict values as attributes and recursive work (backward compatibility)

    return config

def get_config_tune(model_name, dataset_name, config_file='', random_choice=True):
    """
   Get the tune config of given dataset and model from config file.
   Parameters:
       model_name (str): Name of Model.
       dataset_name (str): Name of Dataset.
       config_file (str): Path to Config File. If not specified, Default: "./config/config_train.json"
       random_choice (bool): If True, randomly choose a config from the list of configs.
   Returns:
       config (dict): config of the given dataset and model
   """
    if config_file == '':
        config_file = Path(__file__).parent / "config" / "config_tune.json"
    with open(config_file, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams'][dataset_name] if 'datasetParams' in config_all[model_name] else {}
    model_debug_args = config_all[model_name]['debugParams']
    dataset_args = config_all['datasetCommonParams'][dataset_name]
    # use aligned feature if the model requires it, otherwise use unaligned feature
    dataset_args = dataset_args['aligned'] if (
            model_common_args['need_data_aligned'] and 'aligned' in dataset_args) else dataset_args['unaligned']

    # Random choice of args (finetune needs random initialization)
    if random_choice:
        for item in model_debug_args['d_paras']:
            if type(model_debug_args[item]) == list:
                model_debug_args[item] = random.choice(model_debug_args[item])
            elif type(model_debug_args[item]) == dict: # Nested Params, 2 levels max
                for k, v in model_debug_args[item].items():
                    model_debug_args[item][k] = random.choice(v)

    config = {}
    config['model_name'] = model_name
    config['dataset_name'] = dataset_name
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(model_debug_args)
    config.update(dataset_args)
    config['video_path'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['video_path'])
    config['annotate_path'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['annotate_path'])
    config['feature_path'] = os.path.join(config_all['datasetCommonParams']['dataset_root_dir'], config['feature_path'])

    config = edict(config) # Edict allows access to dict values as attributes and recursive work (backward compatibility)

    return config
