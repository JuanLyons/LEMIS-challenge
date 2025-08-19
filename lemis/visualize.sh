python visualize.py --dataset-path /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/data/sages/frames --coco-ann-path /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/data/sages/annotations/fold1/test_annotation_coco_polygon.json \
--selection-type IoU --n-images 100 --images-per-plot 1 --plot-original-images \
--preds-path /home/jclyons/endovis/challenge_2025/MaskDINO/outputs/cvs-challenge_pretrained_endoscapes_fold1_2_test/inference/coco_instances_results.json \
--preds-name LEMIS --output_path ./visualization_maskdino/fold1 --filter --selection topk_thresh --selection_info 5,0.1

python visualize.py --dataset-path /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/data/sages/frames --coco-ann-path /media/SSD1/scanar/endovis/LEMIS-challenge/region_proposals/data/sages/annotations/fold2/test_annotation_coco_polygon.json \
--selection-type IoU --n-images 100 --images-per-plot 1 --plot-original-images \
--preds-path /home/jclyons/endovis/challenge_2025/MaskDINO/outputs/cvs-challenge_pretrained_endoscapes_fold2_2_test/inference/coco_instances_results.json \
--preds-name LEMIS --output_path ./visualization_maskdino/fold2 --filter --selection topk_thresh --selection_info 5,0.1