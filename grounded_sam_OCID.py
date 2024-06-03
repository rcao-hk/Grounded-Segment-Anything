import argparse
import sys

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import json
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

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


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())
        
    return boxes_filt, torch.Tensor(scores), pred_phrases


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
    unique_labels = unique_labels[unique_labels > 1]  # 去掉背景标签（假设背景为0）

    if unique_labels.size == 0:
        return []
    
    # 遍历每一个物体，计算其bounding box
    for label in unique_labels:
        # 获取当前物体的mask
        object_mask = np.where(mask_map == label, 255, 0).astype(np.uint8)

        # 计算bounding box
        x, y, w, h = cv2.boundingRect(object_mask)

        # 将bounding box添加到列表中
        bounding_boxes.append((x, y, x+w, y+h))

    return bounding_boxes


# parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
# parser.add_argument("--config", type=str, required=True, help="path to config file")
# parser.add_argument(
#     "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
# )
# parser.add_argument(
#     "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
# )
# parser.add_argument(
#     "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
# )
# parser.add_argument(
#     "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
# )
# parser.add_argument(
#     "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
# )
# parser.add_argument("--input_image", type=str, required=True, help="path to image file")
# parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
# parser.add_argument(
#     "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
# )
#
# parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
# parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
#
# parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
# args = parser.parse_args()


# cfg
# config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# grounded_checkpoint = "./groundingdino_swint_ogc.pth"

# config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
# grounded_checkpoint = "./groundingdino_swinb_cogcoor.pth"

# config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
# grounded_checkpoint = "./groundingdino_swinb_tune.pth"

config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
grounded_checkpoint = "./groundingdino_swint_graspnet_tune.pth"

sam_version = "vit_h"
sam_checkpoint = "./sam_vit_h_4b8939.pth"
sam_hq_checkpoint = "./sam_hq_vit_h.pth"
use_sam_hq = True
dataset_root = "/media/gpuadmin/rcao/dataset/OCID"
method_id = 'GDS_v0.3.1'
# text_prompt = "object. animal. fruit. "
# text_prompt = "table objects"
text_prompt = 'object'
gt_box = False
mask_save_root = os.path.join('/media/gpuadmin/rcao/result/uois/ocid', '{}_mask'.format(method_id))
# make dir
os.makedirs(mask_save_root, exist_ok=True)

def read_file(file_path):
    f = open(file_path,"r")
    lines = f.readlines()
    data_list = []
    for line in lines:
        data_list.append(line.strip('\n'))
    return data_list

image_list = read_file(os.path.join(dataset_root, 'data_list.txt'))
box_threshold = 0.27
text_threshold = 0.3
iou_threshold = 0.5
use_nms = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vis_save = True
vis_save_interval = 100
vis_save_root = os.path.join(mask_save_root, 'vis')
os.makedirs(vis_save_root, exist_ok=True)
# initialize SAM

# load model
model = load_model(config_file, grounded_checkpoint, device=device)

if use_sam_hq:
    predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).to(device))
else:
    predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

# load image
for image_idx, image_path in enumerate(tqdm(image_list)):
    image_name = os.path.basename(image_path).split('.')[0]
    image_dir = os.path.join(*os.path.dirname(image_path).split('/')[1:-1])
    image_path = os.path.join(dataset_root, 'data', image_path)
    mask_path = os.path.join(dataset_root, 'data', image_path.replace('rgb', 'label'))
    image_pil, image = load_image(image_path)

    if gt_box:
        gt_mask = np.array(Image.open(mask_path))
        gt_bb = get_bounding_boxes(gt_mask)
        input_bb = torch.tensor(gt_bb)
        pred_phrases = ['object' for i in range(len(input_bb))]
    else:
        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, device=device
        )
            
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        input_bb = boxes_filt.cpu()
        if use_nms:
            # use NMS to handle overlapped boxes
            # print(f"Before NMS: {input_bb.shape[0]} boxes")
            nms_idx = torchvision.ops.nms(input_bb, scores, iou_threshold).numpy().tolist()
            input_bb = input_bb[nms_idx]
            pred_phrases = [pred_phrases[idx] for idx in nms_idx]
            # print(f"After NMS: {input_bb.shape[0]} boxes")
        
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    predictor.set_image(input_image)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_bb, input_image.shape[:2]).to(device)
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
            show_box(box.numpy(), plt.gca(), label)
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

    # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)