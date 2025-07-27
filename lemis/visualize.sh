python visualize.py --dataset-path ../data/LED/frames --coco-ann-path /home/jclyons/endovis/miccai2025/LEMIS/data/EndoVis-2018/annotations/endovis_2018_test.json \
--dataset coco_endovis_2018 --video 10 --selection-type IoU --n-images 100 --images-per-plot 1 --plot-original-images \
--preds-path /home/jclyons/endovis/miccai2025/LEMIS/outputs/EndoVis-2018/exp_LED_segmentation_2/train/best_predictions/best_all_18_preds_instruments.json \
--preds-name LEMIS --output_path ./visualization/outputs_coco_endovis_2018