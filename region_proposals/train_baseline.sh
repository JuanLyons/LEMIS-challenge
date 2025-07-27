python train_net.py --num-gpus 3 \
--config-file configs/cvs/CVS_SwinL_fold1.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data \
OUTPUT_DIR outputs/challenge/CVS_SwinL_fold1 \
MODEL.WEIGHTS /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/model_final_SwinL.pkl \
MODEL.TEXT None

python train_net.py --num-gpus 3 \
--config-file configs/cvs/CVS_SwinL_fold2.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data \
OUTPUT_DIR outputs/challenge/CVS_SwinL_fold2 \
MODEL.WEIGHTS /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/model_final_SwinL.pkl \
MODEL.TEXT None

python train_net.py --num-gpus 3 \
--config-file configs/cvs/CVS_swinS_fold1.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data \
OUTPUT_DIR outputs/challenge/CVS_swinS_fold1 \
MODEL.WEIGHTS /home/scanar/GraSP/TAPIS/region_proposals/weights/model_final_swin_S.pkl \
MODEL.TEXT None

python train_net.py --num-gpus 3 \
--config-file configs/cvs/CVS_swinS_fold2.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data \
OUTPUT_DIR outputs/challenge/CVS_swinS_fold2 \
MODEL.WEIGHTS /home/scanar/GraSP/TAPIS/region_proposals/weights/model_final_swin_S.pkl \
MODEL.TEXT None

python train_net.py --num-gpus 3 \
--config-file configs/cvs/CVS_R50_fold1.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data \
OUTPUT_DIR outputs/challenge/CVS_R50_fold1 \
MODEL.WEIGHTS /home/scanar/GraSP/TAPIS/region_proposals/weights/model_final_r50.pkl \
MODEL.TEXT None

python train_net.py --num-gpus 3 \
--config-file configs/cvs/CVS_R50_fold2.yaml \
DATASETS.DATA_PATH /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data \
OUTPUT_DIR outputs/challenge/CVS_R50_fold2 \
MODEL.WEIGHTS /home/scanar/GraSP/TAPIS/region_proposals/weights/model_final_r50.pkl \
MODEL.TEXT None