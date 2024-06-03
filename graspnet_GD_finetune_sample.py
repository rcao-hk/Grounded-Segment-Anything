import os
from shutil import copyfile

dataset_root = '/media/gpuadmin/rcao/dataset/graspnet'
dataset_save_root = '/media/gpuadmin/rcao/dataset/graspnet_uois'

import json
import numpy as np
from PIL import Image


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


# Function to create a COCO formatted JSON file from a set of images and masks
def create_custom_json(image_path, mask_path, json_file):
    data_list = []

    filename = os.path.basename(json_file)
    # Load image and mask
    image = Image.open(image_path)
    mask = np.array(Image.open(mask_path))

    shapes = []
    for obj_id in np.unique(mask):
        if obj_id == 0:  # skip background
            continue
        obj_mask = (mask == obj_id)
        corner_point = calculate_points(obj_mask)
        # Convert bounding box to top-left and bottom-right corners
        points = [[corner_point[0], corner_point[1]], [corner_point[2], corner_point[3]]]
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


def create_coco_label_json(image_star_idx, anno_start_idx, image_path, mask_path, image_name):
    image = Image.open(image_path)
    mask = np.array(Image.open(mask_path))

    images = [{
        "height": image.height,
        "width": image.width,
        "id": image_star_idx + 1,  # Assuming a static ID for the single image, modify as needed
        "file_name": image_name
    }]

    annotations = []
    for obj_idx, obj_id in enumerate(np.unique(mask)):
        if obj_id == 0:  # Skip background
            continue
        obj_mask = (mask == obj_id)
        bbox = calculate_bbox(obj_mask)
        area = calculate_area(bbox)
        segmentation = calculate_segmentation(bbox)

        annotations.append({
            "iscrowd": 0,
            "category_id": 1,
            "id": int(anno_start_idx + obj_idx),  # Using the object ID from the mask as annotation ID
            "image_id": image_star_idx + 1,  # Assuming a static image ID
            "bbox": bbox,
            "area": area,
            "segmentation": segmentation
        })
    return images, annotations


train_list = list(range(100))
test_list = list(range(100, 190))

images = []
annotations = []
train_images = []
train_annotations = []
test_images = []
test_annotations = []

sample_interval = 25
for camera in ['realsense']:
    label_save_root = os.path.join(dataset_save_root, camera, 'annotations')
    os.makedirs(label_save_root, exist_ok=True)
    for scene_idx in range(190):
        for view_idx in range(256):
            if view_idx % sample_interval != 0:
                continue
            print("camera:{}, scene index:{}, anno index:{}".format(camera, scene_idx, view_idx))

            rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, view_idx))
            mask_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, view_idx))

            rgb_save_root = os.path.join(dataset_save_root, camera, 'images')
            os.makedirs(rgb_save_root, exist_ok=True)
            copyfile(rgb_path, os.path.join(rgb_save_root, 'scene_{:04d}_{:04d}.png'.format(scene_idx, view_idx)))

            mask_save_root = os.path.join(dataset_save_root, camera, 'labels')
            os.makedirs(mask_save_root, exist_ok=True)
            create_custom_json(os.path.join(rgb_save_root, 'scene_{:04d}_{:04d}.png'.format(scene_idx, view_idx)), mask_path,
                                          os.path.join(mask_save_root, 'scene_{:04d}_{:04d}.json'.format(scene_idx, view_idx)))

            scene_images, scene_annotations = create_coco_label_json(len(images), len(annotations),
                                                                     rgb_path, mask_path, 'scene_{:04d}_{:04d}.png'.format(scene_idx, view_idx))
            images.extend(scene_images)
            annotations.extend(scene_annotations)
            if scene_idx in train_list:
                train_images.extend(scene_images)
                train_annotations.extend(scene_annotations)
            elif scene_idx in test_list:
                test_images.extend(scene_images)
                test_annotations.extend(scene_annotations)

    categories = [{"id": 1, "name": "object"}]  # Example category, adjust as necessary
    annotation_all_dict = {
        "images": images,
        "categories": categories,
        "annotations": annotations
    }

    with open(os.path.join(label_save_root, "annotations_all.json"), 'w') as f:
        json.dump(annotation_all_dict, f, indent=2)

    train_dict = {
        "images": train_images,
        "categories": categories,
        "annotations": train_annotations
    }
    with open(os.path.join(label_save_root, "trainval.json"), 'w') as f:
        json.dump(annotation_all_dict, f, indent=2)

    test_dict = {
        "images": test_images,
        "categories": categories,
        "annotations": test_annotations
    }
    with open(os.path.join(label_save_root, "test.json"), 'w') as f:
        json.dump(annotation_all_dict, f, indent=2)