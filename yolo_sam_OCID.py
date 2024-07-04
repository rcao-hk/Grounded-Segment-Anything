import argparse
import sys
import os

import numpy as np
import json
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T

# segment anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


def get_bounding_boxes(mask_map):
    # 初始化一个列表来存储bounding boxes
    bounding_boxes = []

    # 使用np.unique获取所有独立物体的索引（除去背景）
    unique_labels = np.unique(mask_map)
    unique_labels = unique_labels[unique_labels != 0]  # 去掉背景标签（假设背景为0）

    # 遍历每一个物体，计算其bounding box
    for label in unique_labels:
        # 获取当前物体的mask
        object_mask = np.where(mask_map == label, 255, 0).astype(np.uint8)

        # 计算bounding box
        x, y, w, h = cv2.boundingRect(object_mask)

        # 将bounding box添加到列表中
        bounding_boxes.append((x, y, x+w, y+h))

    return bounding_boxes


def yolov8_detection(model, image, args):
    results = model(image, stream=True, conf=args.box_threshold, iou=args.iou_threshold)  # generator of Results objects

    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
    
    bbox = boxes.xyxy.tolist()
    bbox = [[int(i) for i in box] for box in bbox]
    return bbox


parser = argparse.ArgumentParser("YOLO with SAM Demo", add_help=True)
parser.add_argument(
    "--yolo_checkpoint", type=str, default="yolov8s.pt", help="path to checkpoint file"
)
parser.add_argument(
    "--sam_version", type=str, default="vit_h", help="SAM ViT version: vit_b / vit_l / vit_h"
)
parser.add_argument(
    "--sam_checkpoint", type=str, default="./sam_vit_h_4b8939.pth", help="path to sam checkpoint file"
)
parser.add_argument(
    "--sam_hq_checkpoint", type=str, default="./sam_hq_vit_h.pth", help="path to sam-hq checkpoint file"
)
parser.add_argument(
    "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
)
parser.add_argument(
    "--use_use_nms", action="store_true", help="using NMS for prediction"
)
parser.add_argument("--dataset_root", type=str, default="/media/gpuadmin/rcao/dataset/OCID", help="dataset root")
parser.add_argument("--save_vis", action="store_true", help="flag to save visualization")
parser.add_argument("--method_id", type=str, required=True, help="method id")
parser.add_argument("--camera_type", type=str, default='realsense', help="camera type")
parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
parser.add_argument("--text_threshold", type=float, default=0.3, help="text threshold")
parser.add_argument("--iou_threshold", type=float, default=0.3, help="iou threshold for NMS")
args = parser.parse_args()
print(args)

sam_version = args.sam_version
sam_checkpoint = args.sam_checkpoint
sam_hq_checkpoint = args.sam_hq_checkpoint
use_sam_hq = args.use_sam_hq
use_nms = args.use_use_nms
dataset_root = args.dataset_root

gt_box = False
method_id = args.method_id
camera_type = args.camera_type
box_threshold = args.box_threshold
text_threshold = args.text_threshold
iou_threshold = args.iou_threshold

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gt_box = False
mask_save_root = os.path.join('/media/gpuadmin/rcao/result/uois/ocid', '{}_mask'.format(method_id))
# make dir
os.makedirs(mask_save_root, exist_ok=True)
vis_save = args.save_vis
vis_save_interval = 100
vis_save_root = os.path.join(mask_save_root, 'vis')
os.makedirs(vis_save_root, exist_ok=True)

def read_file(file_path):
    f = open(file_path,"r")
    lines = f.readlines()
    data_list = []
    for line in lines:
        data_list.append(line.strip('\n'))
    return data_list

image_list = read_file(os.path.join(dataset_root, 'data_list.txt'))
# initialize YOLO
model = YOLO('yolo_uoais_tuned.pt')
model.to(device)

# initialize SAM
if use_sam_hq:
    predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
else:
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

# load image
# scene_list = trange(130, 160)
for image_idx, image_path in enumerate(tqdm(image_list)):
    image_name = os.path.basename(image_path).split('.')[0]
    image_dir = os.path.join(*os.path.dirname(image_path).split('/')[1:-1])
    image_path = os.path.join(dataset_root, 'data', image_path)
    mask_path = os.path.join(dataset_root, 'data', image_path.replace('rgb', 'label'))
    
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    # if scene_idx== 187 and view_idx >=62 and view_idx <= 221:
    #     gt_box = True
    # else:
    #     gt_box = False
    if gt_box:
        gt_mask = np.array(Image.open(mask_path))
        gt_bb = get_bounding_boxes(gt_mask)
        input_bb = torch.tensor(gt_bb)
        # if view_idx == 64:
        #     import copy
        #     record_bb = copy.deepcopy(input_bb)
        pred_phrases = ['object' for i in range(len(input_bb))]
    else:
        yolov8_boxex = yolov8_detection(model, input_image, args)
        input_bb = torch.tensor(yolov8_boxex, device=predictor.device)
        pred_phrases = ['object' for i in range(len(input_bb))]
    # if view_idx >=64 and view_idx <= 221:
    #     input_bb = record_bb 

    predictor.set_image(input_image)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_bb, input_image.shape[:2]).to(device)
    if len(input_bb) == 0:
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=None,
            multimask_output=False,
        )
    else:
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(device),
            multimask_output=False,
        )

    if vis_save and image_idx % vis_save_interval == 0:
        # draw output image
        plt.figure(figsize=(10, 10))
        plt.imshow(input_image)
        for box, label in zip(input_bb, pred_phrases):
            show_box(box.cpu().numpy(), plt.gca(), label)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

        plt.axis('off')
        plt.savefig(
            os.path.join(vis_save_root, "{}.png".format(image_name)),
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        
    masks = masks.detach().cpu().numpy()
    pred_mask = np.zeros((input_image.shape[0], input_image.shape[1]))
    for inst_id, inst_mask in enumerate(masks):
        pred_mask[inst_mask[0]] = inst_id + 1

    pred_mask = (pred_mask / np.max(pred_mask)) * 255
    result = Image.fromarray(pred_mask.astype(np.uint8))
    mask_save_path = os.path.join(mask_save_root, image_dir)
    os.makedirs(mask_save_path, exist_ok=True)
    result.save(os.path.join(mask_save_path, '{}.png'.format(image_name)))