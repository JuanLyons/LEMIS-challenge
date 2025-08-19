############################################################ Sages fold1

python match_annots_n_preds.py --coco_anns_path /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/data/sages_cutmargins/annotations/fold1/train_annotation_coco_polygon.json \
--preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/region_proposals/outputs/features_fold1_train/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/annotations/train_train_anns.json \
--out_coco_preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/annotations/train_train_preds.json \
--out_features_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/features/train_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/data/sages_cutmargins/annotations/fold1/train_annotation_coco_polygon.json \
--preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/region_proposals/outputs/features_fold1_train/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/annotations/train_val_anns.json \
--out_coco_preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/annotations/train_val_preds.json \
--out_features_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/features/train_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/annotations/test_annotation_coco_polygon.json \
--preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/region_proposals/outputs/features_fold1_test/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/annotations/test_train_anns.json \
--out_coco_preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/annotations/test_train_preds.json \
--out_features_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/features/test_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/annotations/test_annotation_coco_polygon.json \
--preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/region_proposals/outputs/features_fold1_test/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/annotations/test_val_anns.json \
--out_coco_preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/annotations/test_val_preds.json \
--out_features_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold1/features/test_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

############################################################ Sages fold2

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/train_annotation_coco_polygon.json \
--preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/region_proposals/outputs/features_fold2_train/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/train_train_anns.json \
--out_coco_preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/train_train_preds.json \
--out_features_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/features/train_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/train_annotation_coco_polygon.json \
--preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/region_proposals/outputs/features_fold2_train/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/train_val_anns.json \
--out_coco_preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/train_val_preds.json \
--out_features_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/features/train_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/test_annotation_coco_polygon.json \
--preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/region_proposals/outputs/features_fold2_test/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/test_train_anns.json \
--out_coco_preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/test_train_preds.json \
--out_features_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/features/test_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/test_annotation_coco_polygon.json \
--preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/region_proposals/outputs/features_fold2_test/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/test_val_anns.json \
--out_coco_preds_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/annotations/test_val_preds.json \
--out_features_path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/data/sages_fold2/features/test_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out