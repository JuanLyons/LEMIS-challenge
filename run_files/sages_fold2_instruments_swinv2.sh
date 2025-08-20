# Experiment setup
TRAIN_FOLD="train" # or fold1, fold2
TEST_FOLD="test" # or fold1, fold2
EXP_PREFIX="sages_fold2_11" # costumize
TASK="INSTRUMENTS"
ARCH="SwinV2"

#-------------------------
DATASET="SAGES"
EXPERIMENT_NAME=$EXP_PREFIX"/"$TRAIN_FOLD
CONFIG_PATH="configs/"$DATASET"/"$ARCH"_"$TASK".yaml"
OUTPUT_DIR="./outputs/"$DATASET"/"$EXPERIMENT_NAME
DATA="data"

DATASET="sages_fold2"
# Change this variables if data is not located in ./data
FRAME_DIR="./"$DATA"/"$DATASET"/frames"
FRAME_LIST="./"$DATA"/"$DATASET"/frame_lists"
ANNOT_DIR="./"$DATA"/"$DATASET"/annotations/"
COCO_ANN_PATH="./"$DATA"/"$DATASET"/annotations/"$TEST_FOLD"_annotation_coco_polygon.json"
FF_TRAIN="./"$DATA"/"$DATASET"/features/"$TRAIN_FOLD"_train_region_features.pth" 
FF_TEST="./"$DATA"/"$DATASET"/features/"$TEST_FOLD"_val_region_features.pth"
#CHECKPOINT="./"$DATA"/"$DATASET"/pretrained_models/"$TASK".pyth"
CHECKPOINT="./"$DATA"/"$DATASET"/pretrained_models/k400_16.pyth"

#-------------------------
# Run experiment

export PYTHONPATH=/home/jclyons/endovis/miccai2025/LEMIS/lemis:$PYTHONPATH

# # Uncomment to calculate region proposals on the fly
# export export PYTHONPATH=/home/jclyons/endovis/miccai2025/LEMIS/region_proposals:$PYTHONPATH

mkdir -p $OUTPUT_DIR
python -B tools/run_net.py \
--cfg $CONFIG_PATH \
NUM_GPUS 1 \
TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
TRAIN.CHECKPOINT_EPOCH_RESET True \
TEST.ENABLE False \
TRAIN.ENABLE True \
ENDOVIS_DATASET.FRAME_DIR $FRAME_DIR \
ENDOVIS_DATASET.FRAME_LIST_DIR $FRAME_LIST \
ENDOVIS_DATASET.TRAIN_LISTS $TRAIN_FOLD".csv" \
ENDOVIS_DATASET.TEST_LISTS $TEST_FOLD".csv" \
ENDOVIS_DATASET.ANNOTATION_DIR $ANNOT_DIR \
ENDOVIS_DATASET.TEST_COCO_ANNS $COCO_ANN_PATH \
ENDOVIS_DATASET.TRAIN_GT_BOX_JSON $TRAIN_FOLD".json" \
ENDOVIS_DATASET.TRAIN_PREDICT_BOX_JSON $TRAIN_FOLD"_train_preds.json" \
ENDOVIS_DATASET.TEST_GT_BOX_JSON $TEST_FOLD".json" \
ENDOVIS_DATASET.TEST_PREDICT_BOX_JSON $TEST_FOLD"_val_preds.json" \
FEATURES.TRAIN_FEATURES_PATH $FF_TRAIN \
FEATURES.TEST_FEATURES_PATH $FF_TEST \
TRAIN.BATCH_SIZE 74 \
TEST.BATCH_SIZE 74 \
OUTPUT_DIR $OUTPUT_DIR \
FEATURES.USE_RPN False # Switch to True to calculate region proposals on the fly (this makes training slower)