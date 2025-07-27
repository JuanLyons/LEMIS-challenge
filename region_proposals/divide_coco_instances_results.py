import os
import json
from tqdm import tqdm

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Divide coco instances results")
    parser.add_argument("--preds-dir", help="Path json file for predictions")
    parser.add_argument("--annots-dir", help="Path json file for annotations")
    return parser.parse_args()


def load_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def save_json(data, json_path):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved {json_path}")


def main(args):
    preds_dir = args.preds_dir
    annots_dir = args.annots_dir

    assert os.path.exists(
        preds_dir
    ), f"Predictions directory {preds_dir} does not exist."
    assert os.path.exists(
        annots_dir
    ), f"Annotations directory {annots_dir} does not exist."

    preds = load_json(os.path.join(preds_dir))
    annots = load_json(os.path.join(annots_dir))

    json_file_name = os.path.basename(preds_dir)

    dataset_name_dict = {
        "coco_endovis_2017": "endovis_2017",
        "coco_endovis_2018": "endovis_2018",
        "GraSP": "grasp",
        "SAR-RARP50": "sar_rarp50",
    }

    id2dataset = {
        image["id"]: dataset_name_dict[image["dataset"]] for image in annots["images"]
    }

    datasets_dict = {
        "endovis_2017": [],
        "endovis_2018": [],
        "grasp": [],
        "sar_rarp50": [],
    }

    for pred in tqdm(preds):
        image_id = pred["image_id"]
        dataset_name = id2dataset[image_id]
        datasets_dict[dataset_name].append(pred)

    for dataset_name, images in datasets_dict.items():
        if len(images) == 0:
            continue
        save_json(
            images,
            os.path.join(
                os.path.dirname(preds_dir), f"{dataset_name}_{json_file_name}"
            ),
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
