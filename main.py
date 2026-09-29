import argparse
import json
import wandb

from run import CyIN_run

def parse_args():
    parser = argparse.ArgumentParser(description="All adjustable parameters in CyIN")
    parser.add_argument('-m', '--model', type=str, default='cyin', help='Name of Model',
                        choices=['cyin']) # 'gcnet',
    parser.add_argument('-d','--dataset', type=str, default='mosi', help='Name of Dataset',
                        choices=['mosi', 'mosei', 'iemocap6', 'meld'])
    parser.add_argument('--train-mode', type=str, default='regression',
                        choices=['regression', 'recognition'],
                        help='Train Mode as Sentiment Regression & Emotion Recognition. Default: regression')
    # Regression: -3~3; Classification: Negative(0),Neutral(1),Positive(2); Emotion Recognition: Happy....
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Config dict to override arguments in config_file. Ignored in tune mode. Default: None')
    parser.add_argument('--config-file', type=str, default='./config/config_regression.json',
                        help='Path to Config File. Default: "./config/config_regression.json" or "./config/config_recognition.json"')
    parser.add_argument('-e', '--eval', type=bool, default=True,
                        help='Whether to Run in Evaluation. Default: False')
    # parser.add_argument('-t', '--tune', action='store_true',
    parser.add_argument('-t', '--tune', type=bool, default=False,
                        help='Whether to Finetune Hyper-parameters. Default: False')
    parser.add_argument('-tt', '--tune-times', type=int, default=50, # 50
                        help='Number of Times to Finetune Hyper-parameters. Default: 50')
    parser.add_argument('-Ft', '--feature-T', type=str, default='',
                        help='Path to Text/Textual Feature File. Default: ""')
    parser.add_argument('-Fa', '--feature-A', type=str, default='',
                        help='Path to Audio/Acoustic Feature File. Default: ""')
    parser.add_argument('-Fv', '--feature-V', type=str, default='',
                            help='Path to Vision/Visual Feature File. Default: ""')
    parser.add_argument('-s', '--seeds',  nargs='*', type=int, default=[1111, 1112, 1113, 1114, 1115],
                        help='Specify Multiple Times for Multiple Random Seeds. Default: [1111, 1112, 1113, 1114, 1115]')
    parser.add_argument('-n', '--num-workers', type=int, default=4,
                        help='Number of CPU-workers used to Load Data. Default: 4')
    parser.add_argument('-g', '--gpu-ids', nargs='*', type=int, default=[0], #  [0] for debug
                        help='Specify Which GPUs to use. Default: []'
                             'If empty, automatically assign to the most memory-free GPU. Support both single and multiple GPUs.')
    parser.add_argument('-v', '--verbose-stdout', type=int, default=1,
                        help='Verbose Level of stdout. 0 for error, 1 for info, 2 for debug. Default: 1')
    parser.add_argument('--model-save-dir', type=str, default='./saved_models',
                        help='Path to Save Trained Models. Default: "./saved_models"')
    parser.add_argument('--result-save-dir', type=str, default='./results',
                        help='Path to Save csv Results. Default: "./results"')
    parser.add_argument('--log-save-dir', type=str, default='./logs',
                        help='Path to Save Logs Files. Default: "./logs"')

    return parser.parse_args()

if __name__ == '__main__':
    CyIN_run(
        model_name=terminal_args.model,
        dataset_name=terminal_args.dataset,
        train_mode=terminal_args.train_mode,
        config=terminal_args.config,
        config_file=terminal_args.config_file,
        is_eval=terminal_args.eval,
        is_tune=terminal_args.tune,
        tune_times=terminal_args.tune_times,
        feature_T=terminal_args.feature_T,
        feature_A=terminal_args.feature_A,
        feature_V=terminal_args.feature_V,
        seeds=terminal_args.seeds,
        num_workers=terminal_args.num_workers,
        gpu_ids=terminal_args.gpu_ids,
        verbose_level=terminal_args.verbose_stdout,
        model_save_dir=terminal_args.model_save_dir,
        result_save_dir=terminal_args.result_save_dir,
        log_save_dir=terminal_args.log_save_dir
    )