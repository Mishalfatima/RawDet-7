import os
import json
from glob import glob
import argparse
def generate_coco(args):

    data_root = args.data_root
    # === CONFIG ===
    mode = args.mode  # 'train' or 'val'
    data = args.data  # 'ZURICH' or 'PASCAL_RAW' or 'RAOD' or 'RAW_NOD_NIKON' or 'RAW_NOD_SONY'


    if data == 'ZURICH':
        dat = 'zurich'
    elif data == 'PASCAL_RAW':
        dat = 'praw'
    elif data == 'RAOD':
        dat = 'raod'
    elif data == 'RAW_NOD_NIKON':
        dat = 'nikon'
    elif data == 'RAW_NOD_SONY':
        dat = 'sony'
    else:
        raise ValueError("Invalid data type. Choose from 'ZURICH', 'PASCAL_RAW', 'RAOD', 'RAW_NOD_NIKON', or 'RAW_NOD_SONY'.")

    input_dir = os.path.join(data_root,'annotations_normal_json', mode, data)

    if not os.path.exists(os.path.join(data_root, 'annotations_coco')):
        os.makedirs(os.path.join(data_root, 'annotations_coco'))

    output_json = mode + "_" + dat + '.json'

    output_json = os.path.join(data_root, 'annotations_coco', output_json)
    print(f"Input directory: {input_dir}")
    print(f"Output JSON file: {output_json}")

    # === FIXED CATEGORIES ===
    fixed_classes = ('car', 'truck', 'tram', 'person', 'bicycle', 'motorcycle', 'bus')
    category_name_to_id = {name: idx + 1 for idx, name in enumerate(fixed_classes)}

    # === INIT COCO STRUCTURE ===
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx + 1, "name": name, "supercategory": "none"}
            for idx, name in enumerate(fixed_classes)
        ]
    }

    image_id_counter = 1
    annotation_id_counter = 1

    # === GATHER FILES ===
    json_files = glob(os.path.join(input_dir, "*.json"))
    import pdb; pdb.set_trace()

    for file_path in json_files:
        with open(file_path, "r") as f:
            data = json.load(f)

        image_filename = os.path.basename(data["image_path"])
        img_width = data.get("img_width")
        img_height = data.get("img_height")

        # Add image info
        coco_output["images"].append({
            "id": image_id_counter,
            "file_name": image_filename,
            "width": img_width,
            "height": img_height
        })

        # Process annotations
        for ann in data["annotations"]:
            class_name = ann["class_name"]
            if class_name not in category_name_to_id:
                continue  # skip unknown categories

            x1, y1, x2, y2 = ann["bbox"]
            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue  # skip invalid boxes

            # Score filtering
            raw_score = ann.get("score", 1.0)
            if isinstance(raw_score, list):
                score = raw_score[0] if raw_score else 1.0
            elif isinstance(raw_score, (float, int)):
                score = float(raw_score)
            else:
                score = 1.0

            if score < 0.8:
                continue  # Skip low-confidence detections

            coco_output["annotations"].append({
                "id": annotation_id_counter,
                "image_id": image_id_counter,
                "category_id": category_name_to_id[class_name],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": ann.get("segmentation", []),
                "score": score
            })

            annotation_id_counter += 1

        image_id_counter += 1

    # === SAVE OUTPUT ===
    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"COCO-style annotations saved to {output_json} with score filtering (≥ 0.8)")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--data', default= 'PASCAL_RAW', help='ZURICH,PASCAL_RAW,RAOD, RAW_NOD_NIKON, RAW_NOD_SONY')
    parser.add_argument('--mode', default='val', help='train or val')
    parser.add_argument('--data_root', default='./datasets/RawDet-7/', help='path to the dataset root')
    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    generate_coco(args)

if __name__ == "__main__":
    main()

