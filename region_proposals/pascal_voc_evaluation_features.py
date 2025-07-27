# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table

from detectron2.evaluation.evaluator import DatasetEvaluator

from tqdm import tqdm

import sys

sys.path.append("../lemis")
from evaluate.main_eval import get_img_ann_dict
from evaluate.instance_segmentation_eval import eval_segmentation
from evaluate.utils import gather_info, load_json, filter_preds, format_instances


def format_detectron2_output(
    data_dict, preds, selection, selection_params, segmentation=False
):
    data_dict, id2name = gather_info(data_dict)

    for pred in tqdm(preds, desc="Separating preds"):
        if pred["category_id"] > 0 and pred["score"] > 0.0:
            pred["category_id"] -= 1
            data_dict[id2name[pred["image_id"]]["file_name"]]["instances"].append(pred)

    for name in tqdm(data_dict, desc="Filtering preds"):
        if len(data_dict[name]["instances"]):
            image_id = data_dict[name]["instances"][0]["image_id"]
            instances = filter_preds(
                data_dict[name]["instances"], selection, selection_params
            )
            instances = format_instances(
                instances=instances,
                width=id2name[image_id]["width"],
                height=id2name[image_id]["height"],
                segmentation=segmentation,
            )
            data_dict[name]["instances"] = instances
        else:
            data_dict[name]["instances"] = []

    return data_dict


class PascalVocEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using PASCAL VOC's metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        selection="topk_thresh",
        max_dets_per_image=5,
        min_score=0.1,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file that can be loaded with `torch.load` and
                   contains all the results in the format they are produced by the model.
                2. "coco_instances_results.json" a json file in COCO's result format.
            selection (str): optional, Prediction filtering method.
            max_dets_per_image (int): optional, limit on the maximum number of detections per image.
                By default in COCO, this limit is to 100, but this can be customized
                to be greater, as is needed in evaluation metrics AP fixed and AP pool
                (see https://arxiv.org/pdf/2102.01066.pdf)
                This doesn't affect keypoint evaluation.
            min_score (float): optional, minimum score for a prediction to be considered valid.
        """
        self._dataset_name = dataset_name
        self._logger = logging.getLogger("mask2former.PascalVocEvaluator")
        self._distributed = distributed
        self._output_dir = output_dir
        self._selection = selection
        self._max_dets_per_image = max_dets_per_image
        self._min_score = min_score

        self._tasks = ["segm"]

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        self._annotations = load_json(json_file)

        self._datasets = []
        json_format = {
            "images": [],
            "annotations": [],
            "categories": copy.deepcopy(self._annotations["categories"]),
        }
        annotations = []
        name2dataset = {}

        for image in self._annotations["images"]:
            if image.get("dataset", dataset_name) not in self._datasets:
                self._datasets.append(image.get("dataset", dataset_name))
                annotations.append(copy.deepcopy(json_format))
            dataset_index = self._datasets.index(image.get("dataset", dataset_name))
            annotations[dataset_index]["images"].append(image)
            name2dataset[image["file_name"]] = image.get("dataset", dataset_name)

        for annotation in self._annotations["annotations"]:
            assert (
                annotation["image_name"] in name2dataset
            ), f"Image {annotation['image_name']} not found in annotations."
            dataset = name2dataset[annotation["image_name"]]
            assert (
                dataset in self._datasets
            ), f"Dataset {dataset} not found in datasets {self._datasets}."
            dataset_index = self._datasets.index(dataset)
            annotations[dataset_index]["annotations"].append(annotation)

        self._annotations = copy.deepcopy(annotations)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {
                "image_id": input["image_id"],
                "dataset": input.get("dataset", self._dataset_name),
            }

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"]
                )
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")

        # Divide predictions per dataset
        predictions_datasets = [[] for _ in range(len(self._datasets))]
        for prediction in predictions:
            dataset_index = self._datasets.index(prediction["dataset"])
            predictions_datasets[dataset_index].append(prediction)

        assert len(predictions) == len(
            list(itertools.chain(*predictions_datasets))
        ), "The number of predictions does not match the number of predictions per dataset."
        del predictions

        coco_results = [
            list(itertools.chain(*[x["instances"] for x in predictions]))
            for predictions in predictions_datasets
        ]
        tasks = self._tasks

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = (
                self._metadata.thing_dataset_id_to_contiguous_id
            )
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert (
                min(all_contiguous_ids) == 0
                and max(all_contiguous_ids) == num_classes - 1
            )

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for coco_results_per_dataset in coco_results:
                for result in coco_results_per_dataset:
                    category_id = result["category_id"]
                    assert category_id < num_classes, (
                        f"A prediction has class={category_id}, "
                        f"but the dataset only has {num_classes} classes and "
                        f"predicted class id should be in [0, {num_classes - 1}]."
                    )
                    result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            coco_results_json = list(itertools.chain(*coco_results))
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results_json))
                f.flush()

        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                self._evaluate_predictions_on_coco(
                    coco_results,
                    task,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(coco_eval, task)
            self._results[task] = res

    def _evaluate_predictions_on_coco(self, coco_results, iou_type):
        """
        Evaluate the coco results using COCOEval API.
        """
        assert len(coco_results) > 0

        coco_eval = []

        for idx, (coco_gt_per_dataset, coco_results_per_dataset) in enumerate(
            zip(self._annotations, coco_results)
        ):
            self._logger.info(f"Evaluating dataset: {self._datasets[idx]}")
            coco_results_per_dataset = format_detectron2_output(
                coco_gt_per_dataset,
                coco_results_per_dataset,
                selection=self._selection,
                selection_params=[self._max_dets_per_image, self._min_score],
                segmentation=(iou_type == "segm"),
            )
            task = "instruments" if iou_type == "segm" else None
            img_ann_dict = get_img_ann_dict(coco_gt_per_dataset, task)
            coco_eval_ind, coco_eval_ind_cat = eval_segmentation(
                task, coco_gt_per_dataset, coco_results_per_dataset, img_ann_dict
            )
            coco_eval_ind = {"mAP@0.5IoU_segm": coco_eval_ind}
            coco_eval_ind.update(coco_eval_ind_cat)
            coco_eval_ind = {
                key: round(float(value), 6) for key, value in coco_eval_ind.items()
            }
            coco_eval.append(coco_eval_ind)

        return coco_eval

    def _derive_coco_results(self, coco_eval, iou_type):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str): the type of evaluation, e.g., "segm" for instance segmentation.

        Returns:
            a dict of {metric name: score}
        """
        metric = "mAP@0.5IoU_segm" if iou_type == "segm" else None
        for idx, coco_eval_ind in enumerate(coco_eval):
            # Display the dataset name
            self._logger.info(f"Evaluation results for dataset: {self._datasets[idx]}")

            if coco_eval_ind is None:
                self._logger.warn("No predictions from the model!")
                assert 0

            # the standard metrics
            self._logger.info(
                "Evaluation results for {}: \n".format(iou_type)
                + create_small_table({metric: coco_eval_ind[metric] * 100})
            )
            if not np.isfinite(sum(coco_eval_ind.values())):
                self._logger.info(
                    "Some metrics cannot be computed and is shown as NaN."
                )

            results_per_category = []
            for idx, name in enumerate(coco_eval_ind):
                if name != metric:
                    ap = coco_eval_ind[name]
                    results_per_category.append(("{}".format(name), float(ap * 100)))

            # tabulate it
            N_COLS = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            results_2d = itertools.zip_longest(
                *[results_flatten[i::N_COLS] for i in range(N_COLS)]
            )
            table = tabulate(
                results_2d,
                tablefmt="pipe",
                floatfmt=".3f",
                headers=["category", "AP"] * (N_COLS // 2),
                numalign="left",
            )
            self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results = round(
            float(np.mean([result["mAP@0.5IoU_segm"] for result in coco_eval])) * 100, 6
        )
        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    try:
        boxes = instances.pred_boxes.tensor.numpy()
        boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        has_embd = False
        if instances.has("mask_embd"):
            embeds = instances.mask_embd.tolist()
            has_embd = True

        has_decod_out = False
        if instances.has("decod_out"):
            dec_outs = instances.decod_out.tolist()
            has_decod_out = True

        has_score_dist = False
        if instances.has("score_dist"):
            score_dists = instances.score_dist.tolist()
            has_score_dist = True
    except:
        breakpoint()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_embd:
            result["mask_embd"] = embeds[k]
        if has_decod_out:
            result["decoder_out"] = dec_outs[k]
        if has_score_dist:
            result["score_dist"] = score_dists[k]
        results.append(result)
    return results
