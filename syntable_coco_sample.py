import os
from shutil import copyfile
from collections import defaultdict

dataset_root = '/media/user/4TB-1/dataset/syntable'
dataset_save_root = '/media/user/4TB-1/dataset/syntable_coco_sample'

import json
import numpy as np
from PIL import Image
import glob

# Helper function to calculate bounding boxes from masks
def calculate_points(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [float(cmin), float(rmin), float(cmax), float(rmax)]


def calculate_bbox(mask):
    """ Calculate the bounding box of a binary mask as [x_min, y_min, width, height]. """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def calculate_area(bbox):
    """ Calculate the area of the bounding box. """
    return bbox[2] * bbox[3]


def calculate_segmentation(bbox):
    """ Calculate the segmentation from the bounding box [x_min, y_min, width, height]. """
    x_min, y_min, width, height = bbox
    return [[x_min, y_min, x_min, y_min + height, x_min + width, y_min + height, x_min + width, y_min]]

def bbox_to_polygon(bbox):
    # x_min, y_min, width, height = bbox
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    x_min, y_min, width, height = float(x_min), float(y_min), float(width), float(height)
    return [
        (x_min, y_min),  # 左上角
        (x_min + width, y_min + height),  # 右下角
    ]
    
# Function to create a COCO formatted JSON file from a set of images and masks
def create_custom_json(image_path, image_anns, json_file):
    data_list = []
    image = Image.open(image_path)

    shapes = []
    for ann in image_anns:
        bbox = ann['visible_bbox']
        corner_point = bbox_to_polygon(bbox)
        # Convert bounding box to top-left and bottom-right corners
        points = [corner_point[0], corner_point[1]]
        shapes.append({
            "label": "object",  # Assuming all objects have the same label, adjust as necessary
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        })

    data = {
        "version": "0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.relpath(image_path, start=os.path.dirname(json_file)),
        "imageData": None,
        "imageHeight": image.height,
        "imageWidth": image.width
    }
    data_list.append(data)

    # Save to JSON
    with open(json_file, 'w') as f:
        json.dump(data_list, f, indent=2)


def create_coco_label_json(image_star_idx, image_path, image_anns, image_name):
    image = Image.open(image_path)

    images = [{
        "height": image.height,
        "width": image.width,
        "id": image_star_idx + 1,  # Assuming a static ID for the single image, modify as needed
        "file_name": image_name
    }]

    annotations = []
    for ann in image_anns:
        bbox = ann['visible_bbox']
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        area = ann['area']
        segmentation = ann['visible_mask']
        id = ann['id']
        annotations.append({
            "iscrowd": 0,
            "category_id": 1,
            "id": id,  # Using the object ID from the mask as annotation ID
            "image_id": image_star_idx + 1,  # Assuming a static image ID
            "bbox": bbox,
            "area": area,
            "segmentation": segmentation
        })
    return images, annotations

images = []
annotations = []
train_images = []
train_annotations = []
val_images = []
val_annotations = []

# scene = 'tabletop'
sample_interval = 25

json_save_root = os.path.join(dataset_save_root, 'annotations')
os.makedirs(json_save_root, exist_ok=True)
img_save_root = os.path.join(dataset_save_root, 'images')
os.makedirs(img_save_root, exist_ok=True)
label_save_root = os.path.join(dataset_save_root, 'labels')
os.makedirs(label_save_root, exist_ok=True)

for data_split in ['train', 'val']:

    json_file = os.path.join(dataset_root, data_split, 'uoais_{}.json'.format(data_split))
    with open(json_file) as f:
        data = json.load(f)
        
    # Create image dict
    image_dict = {"%g" % x["id"]: x for x in data["images"]}
    # Create image-annotations dict
    anns = defaultdict(list)
    for ann in data["annotations"]:
        anns[ann["image_id"]].append(ann)
        
    for img_idx, image in image_dict.items():
        img_idx = int(img_idx)
        if img_idx % sample_interval != 0:
            continue
        img_path = os.path.join(dataset_root, data_split, image['file_name'])
        # img_split = image['file_name'].split('/')[0]
        img_id = img_path.split('/')[-1].split('.')[0]
        img_ann = anns[img_idx]
        
        copyfile(img_path, os.path.join(img_save_root, '{}_{}.png'.format(data_split, img_id)))
        create_custom_json(os.path.join(img_save_root, '{}_{}.png'.format(data_split, img_id)), 
                           img_ann, os.path.join(label_save_root, '{}_{}.json'.format(data_split, img_id)))

        scene_images, scene_annotations = create_coco_label_json(len(images), img_path, img_ann, 
                                                                 '{}_{}.png'.format(data_split, img_id))
        
        images.extend(scene_images)
        annotations.extend(scene_annotations)
        if data_split == 'train':
            train_images.extend(scene_images)
            train_annotations.extend(scene_annotations)
        elif data_split == 'val':
            val_images.extend(scene_images)
            val_annotations.extend(scene_annotations)
        
categories = [{"id": 1, "name": "object"}]  # Example category, adjust as necessary
annotation_all_dict = {
    "images": images,
    "categories": categories,
    "annotations": annotations
}

with open(os.path.join(json_save_root, "annotations_all.json"), 'w') as f:
    json.dump(annotation_all_dict, f, indent=2)

train_dict = {
    "images": train_images,
    "categories": categories,
    "annotations": train_annotations
}
with open(os.path.join(json_save_root, "train.json"), 'w') as f:
    json.dump(train_dict, f, indent=2)

val_dict = {
    "images": val_images,
    "categories": categories,
    "annotations": val_annotations
}
with open(os.path.join(json_save_root, "val.json"), 'w') as f:
    json.dump(val_dict, f, indent=2)