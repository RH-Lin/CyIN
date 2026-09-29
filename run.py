import gc # garbage collector
import logging
from pathlib import Path
import time
import random

import torch
import pandas as pd
from easydict import EasyDict as edict
import numpy as np

from utils.Functions import setup_seed, assign_gpu, count_parameters
from config import get_config_train, get_config_tune
from data_loader import MMDataLoader
# from data_loader_diag import MMDataLoader # gcnet
from models import ConstructModels
from trains import ConstructTrains

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # Arrang GPU device in PCI_BUS_ID order
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2" # This is crucial for reproducibility (CUDA>=10.2)

logger = logging.getLogger('CyIN')

def _set_logger(log_save_dir, model_name, dataset_name, verbose_level):
    # Base Logger
    log_file_path = Path(
        log_save_dir) / f"{model_name}-{dataset_name}-{time.localtime().tm_year}{time.localtime().tm_mon:02d}" \
                        f"{time.localtime().tm_mday:02d}.log"
                        # f"{time.localtime().tm_mday}{time.localtime().tm_hour}{time.localtime().tm_min}.log"
    logger = logging.getLogger('CyIN')
    if "LOCAL_RANK" in os.environ.keys():
        logger.setLevel(logging.DEBUG if int(os.environ["LOCAL_RANK"]) in [-1,0] else logging.WARN)
    else:
        logger.setLevel(logging.DEBUG)

    # File Handler which record in a .log file
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    if "LOCAL_RANK" in os.environ.keys():
        fh.setLevel(logging.DEBUG if int(os.environ["LOCAL_RANK"]) in [-1,0] else logging.WARN)
    else:
        fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Stream Handler which print on the terminal/console
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    sh = logging.StreamHandler()
    sh.setLevel(stream_level[verbose_level])
    sh_formatter = logging.Formatter('%(name)s - %(message)s')
    sh.setFormatter(sh_formatter)
    logger.addHandler(sh)

    return logger


def CyIN_run(
        model_name, dataset_name, train_mode, config=None, config_file="",  is_eval=False, is_tune=False, tune_times=50,
        feature_T="", feature_A="", feature_V="",
        seeds=[], num_workers=4, gpu_ids=[0],
        verbose_level=1, model_save_dir="", result_save_dir="", log_save_dir=""
):
    """
    Main function for running the CyIN framework.
    Parameters:
        model_name (str): Name of Model
        dataset_name (str): Name of Dataset
        train_mode (str): Train Mode as Sentiment Regression & Emotion Recognition. Default: regression
        config (dict): Config dict. Used to override arguments in config_file. Ignored in tune mode.
        config_file: Path to Config File. If not specified, Default: "./config/config_regression.json" or "./config/config_recognition.json"
        is_eval (bool): Whether to Run Evaluation. Default: False
        is_tune (bool): Whether to Finetune Hyper-parameters. Default: False
        tune_times (int): Number of Times to Finetune Hyper-parameters. Default: 50
        feature_T (str): Path to Text/Textual Feature File. Default: ""
        feature_A (str): Path to Audio/Acoustic Feature File. Default: ""
        feature_V (str): Path to Vision/Visual Feature File. Default: ""
        seeds (list): List of Seeds. Default: [1111, 1112, 1113, 1114, 1115]
        num_workers (int): Number of CPU-workers used to Load Data. Default: 4
        gpu_ids (list): Specify Which GPUs to use. Default: [0]
                        If empty, automatically assign to the most memory-free GPU. Currently only support single GPU.
        verbose_level (int): Verbose Level of stdout. 0 for error, 1 for info, 2 for debug. Default: 1
        model_save_dir (str): Path to Save Trained Models. Default: "./saved_models"
        result_save_dir (str): Path to Save csv Results. Default: "./results"
        log_save_dir (str): Path to Save Logs Files. Default: "./logs"
    """
    # Initialization
    model_name = model_name.lower() # Converts all uppercase characters in a string to lowercase.
    dataset_name = dataset_name.lower()

    if config_file != "":
        config_file = Path(config_file)
    else: # Use Default config_file: "./config/config_regression.json" or "./config/config_recognition.json"
        if train_mode in ("regression", "classification"):
            if is_tune:
                config_file = Path(__file__).parent / "config" / "config_tune.json"
            else:
                config_file = Path(__file__).parent / "config" / "config_regression.json"
        else:
            if is_tune:
                config_file = Path(__file__).parent / "config" / "config_tune_emo.json"
            else:
                config_file = Path(__file__).parent / "config" / "config_recognition.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found!")
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    if model_save_dir == "": # Use Default model_save_dir: "./saved_models"
        model_save_dir = Path(__file__).parent / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if result_save_dir == "": # Use Default result_save_dir: "./results"
        result_save_dir = Path(__file__).parent / "results"
    Path(result_save_dir).mkdir(parents=True, exist_ok=True)
    if log_save_dir == "": # Use Default log_dir: "./logs"
        log_save_dir = Path(__file__).parent / "logs"
    Path(log_save_dir).mkdir(parents=True, exist_ok=True)

    logger = _set_logger(log_save_dir, model_name, dataset_name, verbose_level)

    logger.info("============================ Program Start ============================")

    if is_tune: # Run Finetune
        setup_seed(seeds[0])
        logger.info(f"FineTuning with seed {seeds[0]}")
        initial_args = get_config_tune(model_name, dataset_name, config_file)
        initial_args['model_save_path'] = Path(model_save_dir) / f"{initial_args['model_name']}-{initial_args['dataset_name']}-tune.pth"
        initial_args['device'] = assign_gpu(initial_args, gpu_ids)
        initial_args['train_mode']= train_mode # Backward compatibility. TODO: remove all train_mode in code
        initial_args['feature_T'] = feature_T
        initial_args['feature_A'] = feature_A
        initial_args['feature_V'] = feature_V
        # if config:  # Override some arguments
        #     import ast #, json
        #     initial_args.update(ast.literal_eval(config))

        result_save_dir = Path(result_save_dir) / "tune"
        result_save_dir.mkdir(parents=True,exist_ok=True)
        has_debuged = [] # Save used params setting
        csv_file = result_save_dir / f"{dataset_name}-{model_name}-tune-{train_mode}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
            for r in range(len(df)):
                has_debuged.append([df.loc[r,c] for c in initial_args['d_paras']])

        for i in range(tune_times):
            args = edict(**initial_args)
            random.seed(time.time())
            new_args = get_config_tune(model_name, dataset_name, config_file)
            setup_seed(seeds[0]) #  Fix the origin random.seed(seeds[0]) (Break reproducibility)
            args.update(new_args)
            args['cur_seed'] = i + 1
            logger.info(f"{'-'*24} Finetuning Hyper-parameters Progress with time [{i + 1}/{tune_times}] {'-'*24}")
            logger.info(f"Args: {args}")
            # Check if the current params has been run
            cur_param = [args[k] for k in args['d_paras']]
            if cur_param in has_debuged:
                logger.info(f"Notice: This set of parameters has been run. Skip.")
                time.sleep(1)
                continue
            has_debuged.append(cur_param)
            # Actual Running
            # result = _run(args, num_workers, is_tune)
            # trainer, model_wo_ddp, dataloader = _run_train(args, num_workers, is_tune, is_eval)

            try:
                trainer, model_wo_ddp, dataloader = _run_train(args, num_workers, is_tune, is_eval)
            except:
                logger.info(f'NaN Loss / Error Running')
                continue

            logger.info(f"Finished Training!")

            if (args.distributed == False) or (args.local_rank == 0):
                result = _run_eval(args, trainer, model_wo_ddp, dataloader)
            
                logger.info(f"FineTuning Result for time [{i + 1}/{tune_times}]: {result} ")
                criterions = list(result['final_test_results'].keys())
                if 'Ids' in criterions: criterions.remove('Ids')
                if 'SResults' in criterions: criterions.remove('SResults')
                if 'Features' in criterions: criterions.remove('Features')
                if 'Labels' in criterions: criterions.remove('Labels')
                # Save running results to csv file
                if Path(csv_file).is_file():
                    df2 = pd.read_csv(csv_file)
                else:
                    df2 = pd.DataFrame(columns=[c for c in args['d_paras']] + [c for c in criterions])
                res = [args[d_p] for d_p in args['d_paras']]
                for col in criterions:
                    value = result['final_test_results'][col]
                    res.append(value)
                df2.loc[len(df2)] = res
                df2.to_csv(csv_file, index=None)
                logger.info(f"Running results have saved to {csv_file}!")
    else: # Run Normal Train
        setup_seed(seeds[0]) # should set before assign_gpu for reproducibility
        args = get_config_train(model_name, dataset_name, config_file)
        args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}-train.pth"
        args['device'] = assign_gpu(args, gpu_ids)
        args['train_mode'] = train_mode  # Backward compatibility. TODO: remove all train_mode in code
        args['feature_T'] = feature_T
        args['feature_A'] = feature_A
        args['feature_V'] = feature_V
        if config:  # Override some arguments
            import ast #, json
            args.update(ast.literal_eval(config))
            # args.update(json.loads(config))

        logger.info("Training with args:")
        logger.info(args)
        logger.info(f"Seeds: {seeds}")
        result_save_dir = Path(result_save_dir) / "train"
        result_save_dir.mkdir(parents=True, exist_ok=True)
        model_results = []

        # Run Training
        logger.info(f"{'-' * 24} Progress with seed {seeds[0]} {'-' * 24}")
        trainer, model_wo_ddp, dataloader = _run_train(args, num_workers, is_tune, is_eval)
        logger.info(f"Finished Training!")

        # result = _run(args, num_workers, is_tune)

        if (args.distributed == False) or (args.local_rank == 0):
            # Run Testing for different missing modality situations
            for i, seed in enumerate(seeds):
                # setup_seed(seed)

                # # Run Training
                # logger.info(f"{'-' * 24} Progress with seed {seeds} {'-' * 24}")
                # trainer, model_wo_ddp, dataloader = _run_train(args, num_workers, is_tune, is_eval)
                # logger.info(f"Finished Training!")

                args['cur_seed'] = i + 1
                logger.info(f"{'-' * 12} Testing with seed {seed} in [{i + 1}/{len(seeds)}] {'-' * 12}")
                # try:
                # result =_run(args, num_workers, is_tune)
                result = _run_eval(args, trainer, model_wo_ddp, dataloader)
                # logger.info(f"Training Result for seed {seed}: {result}")
                model_results.append(result)
                # except:
                #     logger.info(f'NaN Loss / Error Running')
                criterions = list(model_results[0]['final_test_results'].keys())
                if 'Ids' in criterions: criterions.remove('Ids')
                if 'SResults' in criterions: criterions.remove('SResults')
                if 'Features' in criterions: criterions.remove('Features')
                if 'Labels' in criterions: criterions.remove('Labels')
                # Save running results to csv file
                csv_file = result_save_dir / f"{dataset_name}-{model_name}-train-{train_mode}.csv"
                if Path(csv_file).is_file():
                    df = pd.read_csv(csv_file)
                else:
                    # df = pd.DataFrame(index=['Model'] + criterions)
                    df = pd.DataFrame(columns=['Model'] + ['Set'] + criterions)

            # for set in ['valid', 'test']:
            for set in ['test']:
                res = [model_name]
                res.append(set.upper())
                for column in criterions:
                    value_list = [c_key['final_' + set + '_results'][column] for c_key in model_results]
                    if value_list != ([] or [None]):
                        mean = round(np.mean(value_list)*100, 2)
                        std = round(np.std(value_list)*100, 2)
                        res.append((mean, std))
                df.loc[len(df)] = res
                df.to_csv(csv_file, index=None)

            if train_mode == 'regression':
                if dataset_name.upper() in ("MOSEI","MOSI"):
                    logger.info(" Seeds:{}\n Set:{} Loss:{}\n"
                        " *** Sentiment Analysis *** \n"
                        " Has0_acc_2:{}\n Has0_F1_score:{}\n Non0_acc_2:{}\n Non0_F1_score:{}\n"
                        " Mult_acc_3:{}\n Mult_acc_5:{}\n Mult_acc_7:{}\n MAE:{}\n Corr:{}".format(
                            seeds, set.upper(), res[-1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10]
                        )
                    )
            elif train_mode == 'recognition':
                if dataset_name.upper() == "MOSEI":
                    logger.info(" Seeds:{}\n Set:{} Loss:{}\n"
                        " *** Emotion(#6) Recognition (W-F1) ***\n"
                        " Happy:{}\n Sad:{}\n Anger:{}\n Surprise:{}\n Disgust:{}\n Fear:{}\n"
                        " Avg. Accuracy:{}\n Avg. Weighted-F1:{}".format(
                        seeds, set.upper(), res[-1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9]
                    ))
                elif dataset_name.upper() == "IEMOCAP6":
                    logger.info(" Seeds:{}\n Set:{} Loss:{}\n"
                        " *** Emotion(#6) Recognition (Acc/W-F1) ***\n"
                        " Happy:{}\n Sad:{}\n Neutral:{}\n Angry:{}\n Excited:{}\n Frustrated:{}\n"
                        " Avg. Accuracy:{}\n Avg. F1:{}".format(
                        seeds, set.upper(), res[-1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9]
                    ))
                elif dataset_name.upper() == "MELD":
                    logger.info(" Seeds:{}\n Set:{} Loss:{}\n"
                        " *** Emotion(#7) Recognition (Acc/W-F1) ***\n"
                        " Neutral:{}\n Surprise:{}\n Fear:{}\n Sad:{}\n Joy:{}\n Disgust:{}\n Anger:{}\n"
                        " Avg. Accuracy:{}\n Avg. F1:{}".format(
                        seeds, set.upper(), res[-1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10]
                    ))

            logger.info(f"Running results have saved to {csv_file}!")

def _run(args, num_workers=4, is_tune=False):
    # load data and models
    dataloader = MMDataLoader(args, num_workers)
    model = ConstructModels(args).to(args['device'])

    logger.info(f"The model has {count_parameters(model)} trainable parameters!")

    # model distributed on multiple GPUs
    model_wo_ddp = model
    if args.distributed == True:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )
        model_wo_ddp = model.module

    # Do Training
    trainer = ConstructTrains().getTrain(args)
    epoch_results = trainer.do_train(model, dataloader, return_epoch_results=True)

    # Load Trained Model & Do Testing (using best epoch)
    assert Path(args['model_save_path']).exists()
    # print(args['model_save_path'])
    model_wo_ddp.load_state_dict(torch.load(args['model_save_path'], map_location='cpu'), strict=True)

    if is_tune:
        # Use valid set to finetune hyper-parameters
        final_valid_results = trainer.do_test(model_wo_ddp, dataloader['valid'], mode='FINAL_VALID', return_sample_results=False)
        # Path(args['model_save_path']).unlink(missing_ok=True) # Delete loaded saved model file
    else:
        final_valid_results = trainer.do_test(model_wo_ddp, dataloader['valid'], mode='FINAL_VALID', return_sample_results=False)
        final_test_results = trainer.do_test(model_wo_ddp, dataloader['test'], mode='FINAL_TEST', return_sample_results=False)

    del model  # Delete object, i.e. loaded model

    if args.distributed == True:
        torch.distributed.barrier()

    gc.collect()
    with torch.cuda.device(args['device']):
        torch.cuda.empty_cache() # Release CUDA memory
    # Garbage collector: returns the total number of objects released by processing circular references
    # https://www.cnblogs.com/franknihao/p/7326849.html
    time.sleep(1)

    if is_tune:
        return {'epoch_results': epoch_results, 'final_test_results': final_test_results}
    else:
        return {'epoch_results': epoch_results, 'final_valid_results': final_valid_results, 'final_test_results': final_test_results}
    
def _run_train(args, num_workers=4, is_tune=False, is_eval=False):
    # load data and models
    dataloader = MMDataLoader(args, num_workers)
    model = ConstructModels(args).to(args['device'])

    logger.info(f"The model has {count_parameters(model)} trainable parameters!")

    # model distributed on multiple GPUs
    model_wo_ddp = model
    if args.distributed == True:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )
        model_wo_ddp = model.module

    # Do Training
    trainer = ConstructTrains().getTrain(args)
    if not is_eval:
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=True)

    del model  # Delete object, i.e. loaded model

    if args.distributed == True:
        torch.distributed.barrier()

    gc.collect()
    with torch.cuda.device(args['device']):
        torch.cuda.empty_cache() # Release CUDA memory
    # Garbage collector: returns the total number of objects released by processing circular references
    # https://www.cnblogs.com/franknihao/p/7326849.html
    time.sleep(1)

    return trainer, model_wo_ddp, dataloader

def _run_eval(args, trainer, model_wo_ddp, dataloader):

    # # Load Trained Model & Do Testing (using best epoch)
    assert Path(args['model_save_path']).exists()
    model_wo_ddp.load_state_dict(torch.load(args['model_save_path'], map_location='cpu'), strict=True)
    logger.info(f"Loaded trained checkpoint from {args['model_save_path']}")

    final_test_results = trainer.do_test(model_wo_ddp, dataloader['test'], mode='FINAL_TEST', return_sample_results=False)

    return {'final_test_results': final_test_results}