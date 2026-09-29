GPU_IDS=0
MODEL_NAME=cyin
DATASET_NAME=mosi # mosi, mosei, iemocap6, meld
TRAIN_MODE=regression # regression, recognition
CONFIG_FILE=AffectiveComputing/CyIN/config/config_regression.json
MODEL_SAVE_DIR=AffectiveComputing/CyIN/saved_models/
LOG_SAVE_DIR=AffectiveComputing/CyIN/logs/

echo "********* Training CyIN model -- $MODEL_NAME on $DATASET_NAME *********"
CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py \
    --gpu-ids 0 \
    --model $MODEL_NAME \
    --dataset $DATASET_NAME \
    --train-mode $TRAIN_MODE \
    --config-file $CONFIG_FILE \
    --model-save-dir $MODEL_SAVE_DIR \
    --log-save-dir $LOG_SAVE_DIR 
