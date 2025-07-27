import os
import cv2
import json
import numpy as np
from glob import glob
from tqdm import tqdm

from copy import deepcopy

from matplotlib import pyplot as plt

import argparse
from evaluate.utils import load_json, read_detectron2_output

import pycocotools.mask as m

import warnings


def hex_to_rgb(hex):
    return tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))


COLOR_HEX_CODE = [
    {
        "id": 1,
        "name": "Bipolar Forceps",
        "supercategory": "Instrument",
        "color": "#23728c",
    },
    {
        "id": 2,
        "name": "Prograsp Forceps",
        "supercategory": "Instrument",
        "color": "#18ae88",
    },
    {
        "id": 3,
        "name": "Large Needle Driver",
        "supercategory": "Instrument",
        "color": "#59e941",
    },
    {
        "id": 4,
        "name": "Vessel Sealer",
        "supercategory": "Instrument",
        "color": "#0281d0",
    },
    {
        "id": 5,
        "name": "Grasping Retractor",
        "supercategory": "Instrument",
        "color": "#37b6ff",
    },
    {
        "id": 6,
        "name": "Monopolar Curved Scissors",
        "supercategory": "Instrument",
        "color": "#8388f4",
    },
    {
        "id": 7,
        "name": "Ultrasound Probe",
        "supercategory": "Instrument",
        "color": "#bda7d9",
    },
    {
        "id": 8,
        "name": "Suction Instrument",
        "supercategory": "Instrument",
        "color": "#d8355f",
    },
    {
        "id": 9,
        "name": "Clip Applier",
        "supercategory": "Instrument",
        "color": "#e69c3c",
    },
    {
        "id": 10,
        "name": "Laparoscopic Instrument",
        "supercategory": "Instrument",
        "color": "#e9c631",
    },
]

COLOR_HEX_CODE = {item["id"]: item["color"][1:] for item in COLOR_HEX_CODE}

COLOR = {
    key: np.array(hex_to_rgb(value), dtype=np.uint8)
    for key, value in COLOR_HEX_CODE.items()
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation parser")
    parser.add_argument(
        "--dataset-path",
        default=None,
        type=str,
        help="Path to dataset folder",
    )
    parser.add_argument(
        "--plot-original-images",
        action="store_true",
        help="Plot original images",
    )
    parser.add_argument(
        "--method-to-plot",
        default="cv2",
        type=str,
        choices=["plt", "cv2"],
        help="Method to plot the images",
    )
    parser.add_argument(
        "--dataset",
        default="coco_endovis_2017",
        type=str,
        required=True,
        choices=["coco_endovis_2017", "coco_endovis_2018", "GraSP", "SAR-RARP50"],
        help="dataset to visualize if the coco annotations are of the complete dataset",
    )
    parser.add_argument(
        "--video",
        default=None,
        type=str,
        help="Number of the video of the dataset to visualize",
    )
    parser.add_argument(
        "--coco-ann-path",
        default=None,
        type=str,
        required=True,
        help="Path to coco anotations",
    )
    parser.add_argument(
        "--preds-path",
        default=None,
        type=str,
        nargs="+",
        required=True,
        help="Path to predictions",
    )
    parser.add_argument(
        "--preds-name",
        default=None,
        type=str,
        nargs="+",
        required=True,
        help="Name of the predictions for the subplot",
    )
    parser.add_argument(
        "--selection-type",
        default="random",
        type=str,
        choices=["random", "IoU"],
        help="How to choose the images in the video to plot",
    )
    parser.add_argument(
        "--images-per-plot",
        default=4,
        type=int,
        help="Number of images to plot per subplot",
    )
    parser.add_argument(
        "--n-images",
        default=3,
        type=int,
        help="Number of images to plot",
    )
    parser.add_argument("--filter", action="store_true", help="Filter predictions")
    parser.add_argument(
        "--selection",
        type=str,
        default="thresh",
        choices=[
            "thresh",  # General threshold filtering
            "topk",  # General top k filtering
            "topk_thresh",  # Threshold and top k filtering
            "cls_thresh",  # Per-class threshold filtering
            "cls_topk",  # Per-class top k filtering
            "cls_topk_thresh",  # Per-class top k and and threshold filtering
            "all",  # No filtering
        ],
        required=False,
        help="Prediction filtering method",
    )
    parser.add_argument(
        "--selection_info",
        help="Hypermarameters to perform filtering",
        required=False,
        default=0.75,
    )
    parser.add_argument(
        "--output_path", default=None, type=str, help="Output directory"
    )

    args = parser.parse_args()

    return args


def filter_preds(args, coco_ann_path, preds):
    segmentation = True
    if args.selection == "thresh":
        selection_params = [None, float(args.selection_info)]
    elif args.selection == "topk":
        selection_params = [int(args.selection_info), None]
    elif args.selection == "topk_thresh":
        assert (
            type(args.selection_info) == str
            and "," in args.selection_info
            and len(args.selection_info.split(",")) == 2
        )
        selection_params = args.selection_info.split(",")
        selection_params[0] = int(selection_params[0])
        selection_params[1] = float(selection_params[1])
    elif "cls" in args.selection:
        assert type(args.selection_info) == str
        assert os.path.isfile(args.selectrion_info)
        with open(args.selection_info, "r") as f:
            selection_params = json.load(f)
    else:
        raise ValueError(f"Incorrect selection type {args.selection}")
    preds = read_detectron2_output(
        coco_ann_path, preds, args.selection, selection_params, segmentation
    )

    return preds


def clean_preds(preds):
    preds_f = {}
    for pred in preds:
        if len(preds[pred]["instances"]) > 0:
            preds_f[pred] = preds[pred]
    return preds_f


def organize_data(coco_ann):
    id2name = {img["id"]: img["file_name"] for img in coco_ann["images"]}
    id2dataset = {img["id"]: img["dataset"] for img in coco_ann["images"]}
    id2width = {img["id"]: img["width"] for img in coco_ann["images"]}
    id2height = {img["id"]: img["height"] for img in coco_ann["images"]}

    coco_ann_f = {}

    for annot in coco_ann["annotations"]:
        img_id = annot["image_id"]
        new_annot = deepcopy(annot)
        new_annot["dataset"] = id2dataset[img_id]
        new_annot["width"] = id2width[img_id]
        new_annot["height"] = id2height[img_id]
        if id2name[img_id] not in coco_ann_f:
            coco_ann_f[id2name[img_id]] = []
        coco_ann_f[id2name[img_id]].append(new_annot)

    return coco_ann_f


def get_mask_ann(anns, height, width):
    """This function returns the mask of the annotations

    Args:
        anns (list): List of annotations
        height (int): Image height
        width (int): Image width

    Returns:
        np.ndarray: Mask of the annotations
    """
    final_mask = np.zeros((height, width), dtype=np.uint8)

    for ann in anns:
        polys = [np.array(p).flatten().tolist() for p in ann["segmentation"]]
        rles = m.frPyObjects(polys, height, width)
        rle = m.merge(rles)
        mask = m.decode(rle).astype(np.uint8)

        if np.sum(np.logical_and(np.where(final_mask != 0, 1, 0), mask)) != 0:
            warnings.warn("Overlapping masks")

        final_mask[np.where(mask == 1)] = ann["category_id"]

    return final_mask


def get_mask_pred(pred, height, width):
    """This function returns the mask of the prediction

    Args:
        pred (dictionary): Prediction
        height (int): Image height
        width (int): Image width

    Returns:
        np.ndarray: Mask of the prediction
    """
    final_mask = np.zeros((height, width, 13), dtype=float)

    for instance in pred["instances"]:
        mask = instance["segment"]
        mask = m.decode(mask).astype("uint8")

        category_id = np.argmax(instance["instruments_score_dist"]) + 1

        positions = np.where(mask)
        final_mask[positions[0], positions[1], category_id] = np.max(
            instance["instruments_score_dist"]
        )

    final_mask = np.argmax(final_mask, axis=-1)
    return final_mask.astype(np.uint8)


def colorize(mask, categories):
    """This function colorizes the mask based on the categories

    Args:
        mask (np.ndarray): Mask to colorize
        categories (list): List of categories

    Returns:
        np.ndarray: Colorized mask
    """
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for cat in categories:
        cat_id = cat["id"]
        color = COLOR[cat_id]

        color_mask[mask == cat_id] = color

    return color_mask


def get_iou(annot_mask, pred_mask):
    classes = set(np.concatenate((np.unique(annot_mask), np.unique(pred_mask))))
    iou_per_class = []
    for cls in classes:
        if cls == 0:
            continue
        annot_mask_cls = np.where(annot_mask == cls, 1, 0)
        pred_mask_cls = np.where(pred_mask == cls, 1, 0)
        inter = np.logical_and(annot_mask_cls, pred_mask_cls).astype(int)
        union = np.logical_or(annot_mask_cls, pred_mask_cls).astype(int)
        iou = np.sum(inter) / np.sum(union)
        iou_per_class.append(iou)
    return np.mean(iou_per_class)


def get_video_format(video):
    """This function returns the video format of the dataset

    Args:
        video (str): Video number

    Returns:
        str: Video format
    """
    return f"video_{int(video):02d}"


def filter_dataset_video(coco_ann, preds_list, dataset, video):
    """This function filters the dataset and the video to visualize

    Args:
        coco_ann (dict): Coco annotations
        preds_list (list): List of predictions
        dataset (str): Dataset to visualize
        video (str): Video to visualize

    Returns:
        dict: Filtered coco annotations
        list: Filtered predictions
    """
    coco_ann_f = {}
    preds_list_f = []

    video = get_video_format(video)

    for i in range(len(preds_list)):
        preds_f = {}
        for img_name in coco_ann:
            if dataset == coco_ann[img_name][0]["dataset"] and video in img_name:
                coco_ann_f[img_name] = coco_ann[img_name]
                preds_f[img_name] = preds_list[i][img_name]

        preds_list_f.append(preds_f)

    return coco_ann_f, preds_list_f


def filter_dataset(coco_ann, preds_list, dataset):
    """This function filters the dataset and the video to visualize

    Args:
        coco_ann (dict): Coco annotations
        preds_list (list): List of predictions
        dataset (str): Dataset to visualize
        video (str): Video to visualize

    Returns:
        dict: Filtered coco annotations
        list: Filtered predictions
    """
    coco_ann_f = {}
    preds_list_f = []

    for i in range(len(preds_list)):
        preds_f = {}
        for img_name in coco_ann:
            if dataset == coco_ann[img_name][0]["dataset"]:
                coco_ann_f[img_name] = coco_ann[img_name]
                preds_f[img_name] = preds_list[i][img_name]

        preds_list_f.append(preds_f)

    return coco_ann_f, preds_list_f


def get_random_keys(args, coco_ann):
    """This function returns the indexes of the images to visualize

    Args:
        args (argparse.Namespace): Arguments
        coco_ann (dict): Coco annotations
        preds (dict): Predictions

    Returns:
        np.ndarray: Indexes of the images to visualize
    """
    keys = np.random.choice(list(coco_ann.keys()), args.n_images, replace=False)

    return keys


def get_iou_keys(args, coco_ann, preds_list):
    """This function returns the indexes of the images to visualize

    Args:
        args (argparse.Namespace): Arguments
        coco_ann (dict): Coco annotations
        preds (dict): Predictions

    Returns:
        np.ndarray: keys of the images to visualize
    """
    len_ann = len(coco_ann)
    keys = []
    ious = np.zeros((len_ann, len(preds_list)))
    pbar = tqdm(total=len_ann, desc="Calculating IoU")
    for i, img_name in enumerate(coco_ann):
        anns = coco_ann[img_name]
        height, width = cv2.imread(os.path.join(args.dataset_path, img_name)).shape[:2]
        ann_mask = get_mask_ann(anns, height, width)

        keys.append(img_name)

        for j, preds in enumerate(preds_list):
            pred = preds[img_name]

            pred_mask = get_mask_pred(pred, height, width)

            iou = get_iou(ann_mask, pred_mask)
            ious[i, j] = iou

        pbar.update(1)

    idxs = np.argsort(np.mean(ious, axis=1))
    final_idx = np.zeros((args.n_images), dtype=int)
    final_ious = np.zeros((args.n_images, len(preds_list)))
    final_idx[0] = idxs[0]
    final_idx[-1] = idxs[-1]
    final_ious[0] = ious[idxs[0]]
    final_ious[-1] = ious[idxs[-1]]
    args.n_images -= 2
    step = 1 / (1 + args.n_images)
    for i, j in zip(range(1, args.n_images + 1), np.arange(step, 1, step) ** 2):
        final_idx[i] = idxs[round(j * len_ann)]
        final_ious[i] = ious[idxs[round(j * len_ann)]]

    final_keys = np.array(keys)[final_idx]

    return final_keys, final_ious


def get_keys(args, coco_ann, preds_list):
    """This function returns the indexes of the images to visualize

    Args:
        args (argparse.Namespace): Arguments
        coco_ann (dict): Coco annotations
        preds_list (dict): Predictions

    Returns:
        np.ndarray: Indexes of the images to visualize
    """
    if args.selection_type == "random":
        keys = get_random_keys(args, coco_ann)
        ious = None
    elif args.selection_type == "IoU":
        keys, ious = get_iou_keys(args, coco_ann, preds_list)
    return keys, ious


def visualize_without_image(args, coco_ann, preds_list, categories):
    keys, ious = get_keys(args, coco_ann, preds_list)

    images_per_plot = args.images_per_plot
    plot_name = "IoU" if ious is not None else "random"

    if args.method_to_plot == "cv2":
        blank_shape = 30
        for index in range(0, len(keys), images_per_plot):
            img_name = keys[index]
            img_path = os.path.join(args.dataset_path, img_name)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            final_img = np.full(
                (
                    height * images_per_plot + blank_shape * (images_per_plot - 1),
                    (len(preds_list) + 1) * width + blank_shape * (len(preds_list)),
                    3,
                ),
                255,
                dtype=np.uint8,
            )
            for i in range(images_per_plot):
                img_name = keys[index + i]
                img_path = os.path.join(args.dataset_path, img_name)
                img = cv2.imread(img_path)
                height, width = img.shape[:2]

                anns = coco_ann[img_name]
                ann_mask = get_mask_ann(anns, height, width)
                ann_mask = cv2.cvtColor(
                    colorize(ann_mask, categories), cv2.COLOR_BGR2RGB
                )

                pred_masks = []
                for preds in preds_list:
                    pred = preds[img_name]
                    pred_mask = get_mask_pred(pred, height, width)
                    pred_masks.append(pred_mask)

                for j, pred_mask in enumerate(pred_masks):
                    pred_masks[j] = cv2.cvtColor(
                        colorize(pred_mask, categories), cv2.COLOR_BGR2RGB
                    )

                final_img[
                    img.shape[0] * i
                    + blank_shape * (i) : img.shape[0] * (i + 1)
                    + blank_shape * (i),
                    : img.shape[1],
                ] = ann_mask

                for j, pred_mask in enumerate(pred_masks):
                    final_img[
                        img.shape[0] * i
                        + blank_shape * (i) : img.shape[0] * (i + 1)
                        + blank_shape * (i),
                        img.shape[1] * (j + 1)
                        + blank_shape * (j + 1) : img.shape[1] * (j + 2)
                        + blank_shape * (j + 1),
                    ] = pred_mask

            saved = cv2.imwrite(
                os.path.join(
                    args.output_path, f"Visualization_{index//images_per_plot}.png"
                ),
                final_img,
            )
            assert saved, "Image not saved"

    elif args.method_to_plot == "plt":
        for index in range(0, len(keys), images_per_plot):
            fig, axes = plt.subplots(
                images_per_plot, len(preds_list) + 1, figsize=(15, 15)
            )
            for ax in axes.flatten():
                ax.axis("off")
            axes[0, 0].set_title(f"Groundtruth\n{keys[index]}")
            for ax_num, pred_name in enumerate(args.preds_name):
                if ious is None:
                    axes[0, ax_num + 1].set_title(f"{pred_name}")
                else:
                    axes[0, ax_num + 1].set_title(
                        f"{pred_name}\nIoU: {ious[index, ax_num]:.4f}"
                    )
            for num_img in range(images_per_plot):
                img_name = keys[index + num_img]
                img_path = os.path.join(args.dataset_path, img_name)
                img = cv2.imread(img_path)
                height, width = img.shape[:2]

                anns = coco_ann[img_name]
                ann_mask = get_mask_ann(anns, height, width)

                pred_masks = []
                for preds in preds_list:
                    pred = preds[img_name]
                    pred_mask = get_mask_pred(pred, height, width)
                    pred_masks.append(pred_mask)

                ann_mask = colorize(ann_mask, categories)

                for i, pred_mask in enumerate(pred_masks):
                    pred_masks[i] = colorize(pred_mask, categories)

                axes[num_img, 0].imshow(ann_mask)
                if num_img > 0:
                    axes[num_img, 0].set_title(f"{img_name}")
                for i, pred_mask in enumerate(pred_masks):
                    if ious is not None and num_img > 0:
                        axes[num_img, i + 1].set_title(
                            f"IoU: {ious[num_img + index, i]:.4f}"
                        )
                    axes[num_img, i + 1].imshow(pred_mask)

            plt.tight_layout()

            fig.savefig(
                os.path.join(args.output_path, f"{plot_name}_{index//images_per_plot}")
            )
            plt.close()


def visualize_with_image(args, coco_ann, preds_list, categories):
    keys, ious = get_keys(args, coco_ann, preds_list)

    images_per_plot = args.images_per_plot
    plot_name = "IoU" if ious is not None else "random"

    if args.method_to_plot == "cv2":
        blank_shape = 30
        for index in range(0, len(keys), images_per_plot):
            img_name = keys[index]
            img_path = os.path.join(args.dataset_path, img_name)
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            final_img = np.full(
                (
                    height * images_per_plot + blank_shape * (images_per_plot - 1),
                    (len(preds_list) + 2) * width + blank_shape * (len(preds_list) + 1),
                    3,
                ),
                255,
                dtype=np.uint8,
            )
            for i in range(images_per_plot):
                img_name = keys[index + i]
                img_path = os.path.join(args.dataset_path, img_name)
                img = cv2.imread(img_path)
                height, width = img.shape[:2]

                anns = coco_ann[img_name]
                ann_mask = get_mask_ann(anns, height, width)
                ann_mask = cv2.cvtColor(
                    colorize(ann_mask, categories), cv2.COLOR_BGR2RGB
                )

                pred_masks = []
                for preds in preds_list:
                    pred = preds[img_name]
                    pred_mask = get_mask_pred(pred, height, width)
                    pred_masks.append(pred_mask)

                for j, pred_mask in enumerate(pred_masks):
                    pred_masks[j] = cv2.cvtColor(
                        colorize(pred_mask, categories), cv2.COLOR_BGR2RGB
                    )

                final_img[
                    img.shape[0] * i
                    + blank_shape * (i) : img.shape[0] * (i + 1)
                    + blank_shape * (i),
                    : img.shape[1],
                ] = img
                final_img[
                    img.shape[0] * i
                    + blank_shape * (i) : img.shape[0] * (i + 1)
                    + blank_shape * (i),
                    img.shape[1] + blank_shape : img.shape[1] * 2 + blank_shape,
                ] = ann_mask

                for j, pred_mask in enumerate(pred_masks):
                    final_img[
                        img.shape[0] * i
                        + blank_shape * (i) : img.shape[0] * (i + 1)
                        + blank_shape * (i),
                        img.shape[1] * (j + 2)
                        + blank_shape * (j + 2) : img.shape[1] * (j + 3)
                        + blank_shape * (j + 2),
                    ] = pred_mask

            saved = cv2.imwrite(
                os.path.join(
                    args.output_path, f"Visualization_{index//images_per_plot}.png"
                ),
                final_img,
            )
            assert saved, "Image not saved"

    elif args.method_to_plot == "plt":
        for index in range(0, len(keys), images_per_plot):
            fig, axes = plt.subplots(
                images_per_plot, len(preds_list) + 2, figsize=(15, 15)
            )
            for ax in axes.flatten():
                ax.axis("off")
            axes[0, 0].set_title(f"Original images\n{keys[index]}")
            axes[0, 1].set_title("Groundtruth")
            for ax_num, pred_name in enumerate(args.preds_name):
                if ious is None:
                    axes[0, ax_num + 2].set_title(f"{pred_name}")
                else:
                    axes[0, ax_num + 2].set_title(
                        f"{pred_name}\nIoU: {ious[index, ax_num]:.4f}"
                    )
            for num_img in range(images_per_plot):
                img_name = keys[index + num_img]
                img_path = os.path.join(args.dataset_path, img_name)
                img = cv2.imread(img_path)
                height, width = img.shape[:2]

                anns = coco_ann[img_name]
                ann_mask = get_mask_ann(anns, height, width)

                pred_masks = []
                for preds in preds_list:
                    pred = preds[img_name]
                    pred_mask = get_mask_pred(pred, height, width)
                    pred_masks.append(pred_mask)

                ann_mask = colorize(ann_mask, categories)

                for i, pred_mask in enumerate(pred_masks):
                    pred_masks[i] = colorize(pred_mask, categories)

                axes[num_img, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[num_img, 1].imshow(ann_mask)
                if num_img > 0:
                    axes[num_img, 0].set_title(f"{img_name}")
                for i, pred_mask in enumerate(pred_masks):
                    if ious is not None and num_img > 0:
                        axes[num_img, i + 2].set_title(
                            f"IoU: {ious[num_img + index, i]:.4f}"
                        )
                    axes[num_img, i + 2].imshow(pred_mask)

            plt.tight_layout()

            fig.savefig(
                os.path.join(args.output_path, f"{plot_name}_{index//images_per_plot}")
            )
            plt.close()


def main():
    args = parse_args()
    if args.selection_type == "IoU":
        assert args.n_images > 1, "Number of images to visualize must be greater than 1"
    else:
        assert args.n_images > 0, "Number of images to visualize must be greater than 0"
    print(args)

    os.makedirs(args.output_path, exist_ok=True)

    # Load coco annotations
    coco_ann_path = args.coco_ann_path
    coco_ann = load_json(coco_ann_path)

    # Check if dataset is provided or coco_ann belong to one dataset. If not, raise an error
    # Check if a video is provided to visualize. If not, raise an error
    assert (
        len(set([ann["file_name"] for ann in coco_ann["images"]])) > 1
        or args.dataset is not None
    ), "Please provide a dataset name"

    categories = coco_ann["categories"]

    # Load predictions
    preds_path = args.preds_path

    preds_list = []
    number_preds = None
    for pred_path in preds_path:
        if args.filter:
            preds = filter_preds(args, coco_ann_path, pred_path)
            if number_preds is None:
                number_preds = len(preds)
            else:
                assert len(preds) == number_preds, "Different number of predictions"
            preds = clean_preds(preds)
            preds_list.append(preds)
        else:
            preds = load_json(pred_path)
            if number_preds is None:
                number_preds = len(preds)
            else:
                assert len(preds) == number_preds, "Different number of predictions"
            preds = clean_preds(preds)
            preds_list.append(preds)

    # Organize data
    coco_ann = organize_data(coco_ann)

    cen = False
    for ann in coco_ann.values():
        cen = args.dataset == ann[0]["dataset"]
        if cen:
            break
    assert cen, "Dataset not found in coco annotations"

    if args.video is not None:
        coco_ann, preds_list = filter_dataset_video(
            coco_ann, preds_list, args.dataset, args.video
        )
    else:
        coco_ann, preds_list = filter_dataset(coco_ann, preds_list, args.dataset)

    # Visualize
    if args.plot_original_images:
        visualize_with_image(args, coco_ann, preds_list, categories)
    else:
        visualize_without_image(args, coco_ann, preds_list, categories)


if __name__ == "__main__":
    main()
