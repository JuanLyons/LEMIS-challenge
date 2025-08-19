python train_net.py --num-gpus 3 \
--config-file configs/cvs/CVS_SwinL_fold1.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data \
OUTPUT_DIR outputs/challenge/CVS_SwinL_fold1_Endoscapes_weights_5 \
MODEL.WEIGHTS /home/scanar/LEMIS-challenge/region_proposals/outputs/challenge/Endoscapes2023_train/model_best.pth \
MODEL.TEXT None

python train_net.py --num-gpus 3 \
--config-file configs/cvs/CVS_SwinL_fold2.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data \
OUTPUT_DIR outputs/challenge/CVS_SwinL_fold2_Endoscapes_weights_5 \
MODEL.WEIGHTS /home/scanar/LEMIS-challenge/region_proposals/outputs/challenge/Endoscapes2023_train/model_best.pth \
MODEL.TEXT None