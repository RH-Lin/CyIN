GPU_IDS=0
MODEL_NAME=cyin
DATASET_NAME=mosi # mosi, mosei, iemocap6, meld
TRAIN_MODE=regression # regression, recognition
CONFIG_FILE=AffectiveComputing/CyIN/config/config_regression.json
MODEL_SAVE_DIR=AffectiveComputing/CyIN/saved_models/
LOG_SAVE_DIR=AffectiveComputing/CyIN/logs/

# testing in fixed protocols with various (T,A,V) # 'TAV' 'T' 'A' 'V' 'TA' 'TV' 'AV'
MODAL_ID_list=('[0,1,2]' '[0]' '[1]' '[2]' '[0,1]' '[0,2]' '[1,2]')
for ((i = 0; i < ${#MODAL_ID_list[@]}; i++))
do
MODAL_ID=${MODAL_ID_list[$i]}
echo "********* Testing with Fixed Protocols -- modal_idx: $MODAL_ID *********"
CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py \
    --gpu-ids 0 \
    --model $MODEL_NAME \
    --dataset $DATASET_NAME \
    --train-mode $TRAIN_MODE \
    --config-file $CONFIG_FILE \
    --model-save-dir $MODEL_SAVE_DIR \
    --log-save-dir $LOG_SAVE_DIR \
    --config "{'miss_protocol': 'fix', 'fix': {'modal_type':$MODAL_ID}, 'random': {'miss_rate':False, 'miss_prob': False},}" \
    --eval true
done

# testing in dataset-level random missing protocols with various Missing Rate
MR_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)
for MR in "${MR_list[@]}"
do
echo "********* Testing with Dataset-level Random Protocols -- missing rate: $MR *********"
CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py \
    --gpu-ids 0 \
    --model $MODEL_NAME \
    --dataset $DATASET_NAME \
    --train-mode $TRAIN_MODE \
    --config-file $CONFIG_FILE \
    --model-save-dir $MODEL_SAVE_DIR \
    --log-save-dir $LOG_SAVE_DIR \
    --config "{'miss_protocol': 'random', 'fix': {'modal_type':[0,1,2]}, 'random': {'miss_rate':$MR, 'miss_prob': False},}" \
    --eval true
done

# testing in instance-level random missing protocols with various Missing Probability
MP_list=(0.1 0.3 0.5 0.7 0.9 1.0)
for MP in "${MP_list[@]}"
do
echo "********* Testing with Instance-level Random Protocols -- missing prob: $MP *********"
CUDA_VISIBLE_DEVICES=$GPU_IDS python main.py \
    --gpu-ids 0 \
    --model $MODEL_NAME \
    --dataset $DATASET_NAME \
    --train-mode $TRAIN_MODE \
    --config-file $CONFIG_FILE \
    --model-save-dir $MODEL_SAVE_DIR \
    --log-save-dir $LOG_SAVE_DIR \
    --config "{'miss_protocol': 'random', 'fix': {'modal_type':[0,1,2]}, 'random': {'miss_rate':False, 'miss_prob': $MP},}" \
    --eval true
done
