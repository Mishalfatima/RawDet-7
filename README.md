# RAWDET-7 🧠📸

A benchmark for object detection on **RAW images**, built on top of [MMDetection](https://github.com/open-mmlab/mmdetection).

---

## 📦 Installation

First, install the environment by following the official [MMDetection installation guide](https://github.com/open-mmlab/mmdetection).

> 💡 This project relies on [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv), which should be installed along with MMDetection.

---

## 📁 Dataset

Download the dataset from (https://data.dws.informatik.uni-mannheim.de/machinelearning/RawDet-7/) and place it in a directory named "datasets".

## 🛠️ Annotation Conversion

Run the following scripts to convert the dataset to COCO-style annotations.
Annotations with a **score < 0.8** are automatically removed.

```bash
python to_coco.py --mode val --data PASCAL_RAW --data_root ./datasets/RawDet-7
python to_coco_combined.py --mode val --data_root ./datasets/RawDet-7
```
## Configs
Currently supports configs for Faster-RCNN, PAA, and RetinaNet.
``` bash
 Faster-RCNN - ./configs/faster_rcnn/faster-rcnn_r50_fpn_1x_RawDet.py
 PAA - configs/paa/paa_r50_fpn_1x_RawDet.py
 RetinaNet - configs/retinanet/retinanet_r50_fpn_1x_RawDet.py
```

## Training
```bash
python tools/train.py \
  --config ./configs/faster_rcnn/faster-rcnn_r50_fpn_1x_RawDet.py \
  --data_root ./datasets/RawDet-7/ \
  --work-dir ./checkpoints/faster_rcnn/ \
  --quant 4 \
  --is_raw \
  --data NEW
```

## Evaluation:
```bash
python tools/test.py \
  --config ./configs/faster_rcnn/faster-rcnn_r50_fpn_1x_RawDet.py \
  --data_root ./datasets/RawDet-7/ \
  --work-dir ./checkpoints/faster_rcnn/ \
  --quant 4 \
  --is_raw \
  --data NEW
```
