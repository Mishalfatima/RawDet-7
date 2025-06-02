import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO

# Path to COCO annotations and images
annotation_file = './datasets/RawDet-7/coco/combined_val.json'
image_dir = './datasets/RawDet-7/combined_sRGB/val/'

# Load COCO annotations
coco = COCO(annotation_file)

# Get all image IDs
image_ids = coco.getImgIds()
# Select one image
image_id = image_ids[0]
image_info = coco.loadImgs(image_id)[0]

# Load image
image_path = os.path.join(image_dir, image_info['data'], image_info['file_name'])
image = Image.open(image_path)

# Load annotations for the image
annotation_ids = coco.getAnnIds(imgIds=image_id)
annotations = coco.loadAnns(annotation_ids)

# Display image and annotations
fig, ax = plt.subplots(1)
ax.imshow(image)

# Draw bounding boxes
for ann in annotations:
    bbox = ann['bbox']  # [x, y, width, height]
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # Optional: show category name
    category = coco.loadCats(ann['category_id'])[0]['name']
    ax.text(bbox[0], bbox[1] - 5, category, color='white',
            bbox=dict(facecolor='red', alpha=0.5))

plt.axis('off')

# Save the figure to a file
output_path = 'output_visualization.png'
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
print(f"Figure saved to {output_path}")

# Optionally display the image
# plt.show()
