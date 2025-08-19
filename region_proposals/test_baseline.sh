python /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/train_net.py --num-gpus 3 --eval-only \
--config-file /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/configs/cvs_cutted/CVS_SwinL_fold2.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/data/sages_cutmargins \
OUTPUT_DIR outputs/features_fold1_train \
MODEL.WEIGHTS /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/outputs/challenge/CVS_SwinL_cutted_from_end2023_cutted_fold1/model_best.pth \
MODEL.TEXT None

python /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/train_net.py --num-gpus 3 --eval-only \
--config-file /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/configs/cvs_cutted/CVS_SwinL_fold1.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/data/sages_cutmargins \
OUTPUT_DIR outputs/features_fold2_train \
MODEL.WEIGHTS /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/outputs/challenge/CVS_SwinL_cutted_from_end2023_cutted_fold2/model_best.pth \
MODEL.TEXT None

python /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/train_net.py --num-gpus 3 --eval-only \
--config-file /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/configs/cvs_cutted/CVS_SwinL_fold1.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/data/sages_cutmargins \
OUTPUT_DIR outputs/features_fold1_test \
MODEL.WEIGHTS /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/outputs/challenge/CVS_SwinL_cutted_from_end2023_cutted_fold1/model_best.pth \
MODEL.TEXT None

python /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/train_net.py --num-gpus 3 --eval-only \
--config-file /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/configs/cvs_cutted/CVS_SwinL_fold2.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/data/sages_cutmargins \
OUTPUT_DIR outputs/features_fold2_test \
MODEL.WEIGHTS /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/outputs/challenge/CVS_SwinL_cutted_from_end2023_cutted_fold2/model_best.pth \
MODEL.TEXT None