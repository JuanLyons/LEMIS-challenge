python train_net.py --num-gpus 6 --eval-only \
--config-file configs/endovis_2017/endovis_2017_SwinL_train.yaml \
DATASETS.DATA_PATH data_1 \
OUTPUT_DIR outputs/LEMIS_endovis_2017_weights_subset_sar_rarp50_2/model_0000111 \
MODEL.WEIGHTS outputs/LEMIS_endovis_2017_weights_subset_sar_rarp50_2/model_0000111.pth \
MODEL.TEXT Dataset_Embeddings

python train_net.py --num-gpus 6 --eval-only \
--config-file configs/endovis_2018/endovis_2018_SwinL.yaml \
DATASETS.DATA_PATH data_1 \
OUTPUT_DIR outputs/LEMIS_endovis_2018_weights_subset_sar_rarp50_2/model_0000339 \
MODEL.WEIGHTS outputs/LEMIS_endovis_2018_weights_subset_sar_rarp50_2/model_0000339.pth \
MODEL.TEXT Dataset_Embeddings

python train_net.py --num-gpus 6 --eval-only \
--config-file configs/grasp/GraSP_SwinL_train.yaml \
DATASETS.DATA_PATH data_1 \
OUTPUT_DIR outputs/LEMIS_grasp_weights_w_o_sar_rarp50/model_0001313 \
MODEL.WEIGHTS outputs/LEMIS_grasp_weights_w_o_sar_rarp50/model_0001313.pth \
MODEL.TEXT Dataset_Embeddings