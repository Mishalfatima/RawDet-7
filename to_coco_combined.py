import os
import json
from glob import glob
import argparse


def generate_coco_combined(args):
    # === CONFIG ===
    mode = args.mode  # 'train' or 'val'
    data_root = args.data_root
    input_dir = os.path.join(data_root, 'annotations_normal_json')
    base_input_dir = os.path.join(input_dir, mode)

    if not os.path.exists(os.path.join(data_root, 'annotations_coco')):
        os.makedirs(os.path.join(data_root, 'annotations_coco'))


    output_json_path = mode+"_combined.json"
    output_json = os.path.join(data_root, 'coco', output_json_path)

    
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

    # === PROCESS ALL SUBFOLDERS ===
    subfolders = ['PASCAL_RAW', 'RAOD', 'RAW_NOD_NIKON', 'RAW_NOD_SONY', 'ZURICH']

    for folder in subfolders:
        input_dir = os.path.join(base_input_dir, folder)
        json_files = glob(os.path.join(input_dir, "*.json"))

        for file_path in json_files:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Get only the image file name
            image_filename = os.path.basename(data["image_path"])
            img_width = data.get("img_width")
            img_height = data.get("img_height")


            s = []
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

                raw_score = ann.get("score", 1.0)
                if isinstance(raw_score, list):
                    score = raw_score[0] if raw_score else 1.0
                elif isinstance(raw_score, (float, int)):
                    score = float(raw_score)
                else:
                    score = 1.0

                if score > 0.8:
                    coco_output["annotations"].append({
                        "id": annotation_id_counter,
                        "image_id": image_id_counter,
                        "category_id": category_name_to_id[class_name],
                        "bbox": [x1, y1, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                        "segmentation": ann.get("segmentation", []),
                        "score": score  # custom addition
                    })

                    annotation_id_counter += 1
                    s.append(1)

                else:
                    s.append(0)

                
            if sum(s) != 0:
                
                # Add image info with 'data' field for subfolder name
                coco_output["images"].append({
                    "id": image_id_counter,
                    "file_name": image_filename,
                    "width": img_width,
                    "height": img_height,
                    "data": folder
                })


                image_id_counter += 1

    # === SAVE OUTPUT ===
    with open(output_json, "w") as f:
        json.dump(coco_output, f, indent=4)

    print(f"COCO-style annotations saved to {output_json}")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate COCO style dataset')
    parser.add_argument('--mode', default='val', help='train or val')
    parser.add_argument('--data_root', default='./datasets/RawDet-7/', help='path to the dataset root')
    args = parser.parse_args()
    return args
    
def main():

    args = parse_args()
    generate_coco_combined(args)

if __name__ == "__main__":
    main()

