python visualize.py --dataset-path /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data/frames --coco-ann-path /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data/annotations/fold1/test_annotation_coco_polygon.json \
--selection-type IoU --n-images 100 --images-per-plot 1 --plot-original-images \
--preds-path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/region_proposals/outputs/challenge/CVS_SwinL_fold1/inference/coco_instances_results.json \
--preds-name LEMIS --output_path ./visualization/fold1 --filter --selection topk_thresh --selection_info 5,0.1

python visualize.py --dataset-path /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data/frames --coco-ann-path /media/SSD1/scanar/GraSP/TAPIS/region_proposals/data/annotations/fold2/test_annotation_coco_polygon.json \
--selection-type IoU --n-images 100 --images-per-plot 1 --plot-original-images \
--preds-path /home/jclyons/endovis/challenge_2025/LEMIS-challenge/region_proposals/outputs/challenge/CVS_SwinL_fold2/inference/coco_instances_results.json \
--preds-name LEMIS --output_path ./visualization/fold2 --filter --selection topk_thresh --selection_info 5,0.1