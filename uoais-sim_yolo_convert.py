# borrow from https://github.com/ultralytics/JSON2YOLO

import contextlib
import json
from collections import defaultdict

import cv2
import pandas as pd
from PIL import Image

import glob
import os
import shutil
from pathlib import Path
import numpy as np
from PIL import ExifTags
from tqdm import tqdm
from pycocotools import mask

# from utils import *

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break
    
    
def exif_size(img):
    """Returns the EXIF-corrected PIL image size as a tuple (width, height)."""
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def split_rows_simple(file="../data/sm4/out.txt"):  # from utils import *; split_rows_simple()
    """Splits a text file into train, test, and val files based on specified ratios; expects a file path as input."""
    with open(file) as f:
        lines = f.readlines()

    s = Path(file).suffix
    lines = sorted(list(filter(lambda x: len(x) > 0, lines)))
    i, j, k = split_indices(lines, train=0.9, test=0.1, validate=0.0)
    for k, v in {"train": i, "test": j, "val": k}.items():  # key, value pairs
        if v.any():
            new_file = file.replace(s, f"_{k}{s}")
            with open(new_file, "w") as f:
                f.writelines([lines[i] for i in v])
                
                
def split_files(out_path, file_name, prefix_path=""):  # split training data
    """Splits file names into separate train, test, and val datasets and writes them to prefixed paths."""
    file_name = list(filter(lambda x: len(x) > 0, file_name))
    file_name = sorted(file_name)
    i, j, k = split_indices(file_name, train=0.9, test=0.1, validate=0.0)
    datasets = {"train": i, "test": j, "val": k}
    for key, item in datasets.items():
        if item.any():
            with open(f"{out_path}_{key}.txt", "a") as file:
                for i in item:
                    file.write("%s%s\n" % (prefix_path, file_name[i]))
                    

def split_indices(x, train=0.9, test=0.1, validate=0.0, shuffle=True):  # split training data
    """Splits array indices for train, test, and validate datasets according to specified ratios."""
    n = len(x)
    v = np.arange(n)
    if shuffle:
        np.random.shuffle(v)

    i = round(n * train)  # train
    j = round(n * test) + i  # test
    k = round(n * validate) + j  # validate
    return v[:i], v[i:j], v[j:k]  # return indices

             
def make_dirs(dir="new_dir/"):
    """Creates a directory with subdirectories 'labels' and 'images', removing existing ones."""
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)  # delete dir
    for p in dir, dir / "labels", dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir


def write_data_data(fname="data.data", nc=80):
    """Writes a Darknet-style .data file with dataset and training configuration."""
    lines = [
        "classes = %g\n" % nc,
        "train =../out/data_train.txt\n",
        "valid =../out/data_test.txt\n",
        "names =../out/data.names\n",
        "backup = backup/\n",
        "eval = coco\n",
    ]

    with open(fname, "a") as f:
        f.writelines(lines)
        

def convert_uoais_sim_coco_json(data_dir="../coco/annotations/", save_dir='', scene='tabletop', use_segments=False):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    json_dir = os.path.join(data_dir, "annotations")  # json directory
    save_dir = make_dirs(save_dir)  # output directory

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")  
        json_name = os.path.basename(json_file)
        # folder name
        fn.mkdir()
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images["%g" % img_id]
            h, w, f = img["height"], img["width"], img["file_name"]
            if f.split('/')[0] != scene:
                continue
            f_name = f.split('/')[-1]
            
            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue 
                # The COCO box format is [top left x, top left y, width, height]
                # box = np.array(ann["bbox"], dtype=np.float64) 
                box = np.array(ann["visible_bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                # Segments
                if use_segments:
                    rle_to_polygons_holes = False
                    save_rle_masks = False
                    set_coco_segments(rle_to_polygons_holes, save_rle_masks, w, h, f, fn, ann, cls, segments)

            # Copy file and Write label
            load_root = os.path.join(data_dir, json_name.split('.')[0])
            os.makedirs(load_root, exist_ok=True)
            save_root = os.path.join(save_dir, 'images', json_name.split('.')[0])
            os.makedirs(save_root, exist_ok=True)
            shutil.copyfile(os.path.join(load_root, f), os.path.join(save_root, f_name))
            with open((fn / f_name).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(segments[i] if use_segments else bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")


def set_coco_segments(rle_to_polygons_holes, save_rle_masks, w, h, f, fn, ann, cls, segments):
    if 'segmentation' not in ann:
        segments.append([])
        return
    if len(ann['segmentation']) == 0:
        segments.append([])
        return
    if isinstance(ann['segmentation'], dict):
        file_name = f.split('.')[0]
        file_name = file_name + '_' + str(len(segments)) + '.png'
        mask_path = (fn / file_name)
        ann['segmentation'] = rle2polygon(ann['segmentation'], rle_to_polygons_holes, save_rle_masks, mask_path)
        if len(ann['segmentation']) == 0:
            segments.append([])
            return
    if len(ann['segmentation']) > 1:
        s = merge_multi_segment(ann['segmentation'])
        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
    else:
        s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
    s = [cls] + s
    segments.append(s)
    

def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = contour[i]
        if i < num - 1:
            p2 = contour[i + 1]
        else:
            p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1])
    return value < 0


def merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in list(range(0, idx1 + 1)):
        contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
        contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
        contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
        contour.append(contour1[i])
    contour = np.array(contour)
    return contour


def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)


def mask2polygon_external(image):
    contours, hierarchies = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if len(contours) == 0:
        return []
    contours_approx = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)
    polygons = []
    for contour in contours_approx:
        if len(contour) >= 3:
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    return polygons 


def get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2)
            if distance_min < 0:
                distance_min = distance
                idx1 = i
                idx2 = j
            elif distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2


def mask2polygon_holes(image):
    contours, hierarchies = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    if len(contours) == 0:
        return []
    contours_approx = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)
    contours_parent = []
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx < 0 and len(contour) >= 3:
            contours_parent.append(contour)
        else:
            contours_parent.append([])
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]
            if len(contour_parent) == 0:
                continue
            contours_parent[parent_idx] = merge_with_parent(contour_parent, contour)
    contours_parent_tmp = []
    for contour in contours_parent:
        if len(contour) == 0:
            continue
        contours_parent_tmp.append(contour)
    polygons = []
    for contour in contours_parent_tmp:
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    return polygons 


def rle2polygon(segmentation, rle_to_polygons_holes, save_rle_masks, mask_path):
    if isinstance(segmentation["counts"], list):
        segmentation = mask.frPyObjects(segmentation, *segmentation["size"])
    m = mask.decode(segmentation) 
    m[m > 0] = 255
    if save_rle_masks:
        import PIL.Image
        pil_image = PIL.Image.fromarray(m)
        pil_image.save(mask_path)
    if rle_to_polygons_holes:
        polygons = mask2polygon_holes(m)
    else:
        polygons = mask2polygon_external(m)
    return polygons


def min_index(arr1, arr2):
    """
    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


if __name__ == "__main__":

    convert_uoais_sim_coco_json(
        "/media/gpuadmin/rcao/dataset/UOAIS-Sim",  # directory with *.json
        "/media/gpuadmin/rcao/dataset/uoais_sim_yolo",
        scene='tabletop',  # 'bin' or 'tabletop'
        use_segments=False,
    )

    # zip results
    # os.system('zip -r ../coco.zip ../coco')