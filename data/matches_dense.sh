############################################################### LEMIS Test LED

# python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/train.json \
# --preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_subset_sar_rarp50/train_features/inference/instances_predictions.pth \
# --out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/train_train_anns.json \
# --out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/train_train_preds.json \
# --out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/features/train_train_region_features.pth \
# --selection all --segmentation --features_key decoder_out

# python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/train.json \
# --preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_subset_sar_rarp50/train_features/inference/instances_predictions.pth \
# --out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/train_val_anns.json \
# --out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/train_val_preds.json \
# --out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/features/train_val_region_features.pth \
# --selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

# python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/test.json \
# --preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_subset_sar_rarp50/test_features/inference/instances_predictions.pth \
# --out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/test_train_anns.json \
# --out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/test_train_preds.json \
# --out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/features/test_train_region_features.pth \
# --selection all --segmentation --features_key decoder_out

# python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/test.json \
# --preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_subset_sar_rarp50/test_features/inference/instances_predictions.pth \
# --out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/test_val_anns.json \
# --out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/annotations/test_val_preds.json \
# --out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/LED/features/test_val_region_features.pth \
# --selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

############################################################ Endovis 2017

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/endovis_2017_train.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_endovis_2017_weights_subset_sar_rarp50_2/train_features/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/train_train_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/train_train_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/features/train_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/endovis_2017_train.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_endovis_2017_weights_subset_sar_rarp50_2/train_features/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/train_val_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/train_val_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/features/train_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/endovis_2017_test.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_endovis_2017_weights_subset_sar_rarp50_2/test_features/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/test_train_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/test_train_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/features/test_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/endovis_2017_test.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_endovis_2017_weights_subset_sar_rarp50_2/test_features/inference/instances_predictions.pth \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/test_val_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/annotations/test_val_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2017/features/test_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

############################################################ Endovis 2018

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/endovis_2018_train.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_endovis_2018_weights_subset_sar_rarp50_2/train_features/inference/coco_instances_results.json \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/train_train_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/train_train_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/features/train_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/endovis_2018_train.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_endovis_2018_weights_subset_sar_rarp50_2/train_features/inference/coco_instances_results.json \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/train_val_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/train_val_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/features/train_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/endovis_2018_test.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_endovis_2018_weights_subset_sar_rarp50_2/test_features/inference/coco_instances_results.json \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/test_train_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/test_train_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/features/test_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/endovis_2018_test.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_endovis_2018_weights_subset_sar_rarp50_2/test_features/inference/coco_instances_results.json \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/test_val_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/test_val_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/features/test_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

############################################################ GraSP

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/grasp_train.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_grasp_weights_subset_sar_rarp50/train_features/inference/coco_instances_results.json \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/train_train_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/train_train_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/features/train_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/grasp_train.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_grasp_weights_subset_sar_rarp50/train_features/inference/coco_instances_results.json \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/train_val_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/train_val_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/features/train_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/grasp_test.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_grasp_weights_subset_sar_rarp50/test_features/inference/coco_instances_results.json \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/test_train_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/test_train_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/features/test_train_region_features.pth \
--selection all --segmentation --features_key decoder_out

python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/grasp_test.json \
--preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_grasp_weights_subset_sar_rarp50/test_features/inference/coco_instances_results.json \
--out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/test_val_anns.json \
--out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/annotations/test_val_preds.json \
--out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/GraSP/features/test_val_region_features.pth \
--selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

############################################################ SAR-RARP50

# python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/sar_rarp50_train.json \
# --preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_sar_rarp50/train_features/inference/instances_predictions.pth \
# --out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/train_train_anns.json \
# --out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/train_train_preds.json \
# --out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/features/train_train_region_features.pth \
# --selection all --segmentation --features_key decoder_out

# python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/sar_rarp50_train.json \
# --preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_sar_rarp50/train_features/inference/instances_predictions.pth \
# --out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/train_val_anns.json \
# --out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/train_val_preds.json \
# --out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/features/train_val_region_features.pth \
# --selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out

# python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/sar_rarp50_test.json \
# --preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_sar_rarp50/test_features/inference/instances_predictions.pth \
# --out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/test_train_anns.json \
# --out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/test_train_preds.json \
# --out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/features/test_train_region_features.pth \
# --selection all --segmentation --features_key decoder_out

# python match_annots_n_preds.py --coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/sar_rarp50_test.json \
# --preds_path /home/jclyons/endovis/miccai2025/LEMIS/region_proposals/outputs/LEMIS_sar_rarp50/test_features/inference/instances_predictions.pth \
# --out_coco_anns_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/test_val_anns.json \
# --out_coco_preds_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/annotations/test_val_preds.json \
# --out_features_path /home/jclyons/endovis/miccai2025/LEMIS/data/SAR-RARP50/features/test_val_region_features.pth \
# --selection topk_thresh --selection_info 5,0.1 --segmentation --validation --features_key decoder_out