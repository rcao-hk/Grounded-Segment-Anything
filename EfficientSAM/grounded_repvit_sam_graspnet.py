import cv2
import numpy as np
import supervision as sv

from tqdm import trange

import os
import torch
import torchvision
from PIL import Image

from groundingdino.util.inference import Model
from segment_anything import SamPredictor
from RepViTSAM.setup_repvit_sam import build_sam_repvit

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "../groundingdino_swint_ogc.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building MobileSAM predictor
RepViTSAM_CHECKPOINT_PATH = "repvit_sam.pt"
repvit_sam = build_sam_repvit(checkpoint=RepViTSAM_CHECKPOINT_PATH)
repvit_sam.to(device=DEVICE)

sam_predictor = SamPredictor(repvit_sam)


# Predict classes and hyper-param for GroundingDINO
dataset_root = '/media/rcao/Data/Dataset/graspnet'
mask_save_root = os.path.join(dataset_root, 'GDS_mask')

camera_type = 'realsense'

# CLASSES = ['object', 'animal', 'tool', ]
CLASSES = ["object", "animal", "tool", "body", "cylinder"]
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


scene_idx = 100
for view_idx in trange(256):
    image_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/realsense/rgb/{:04d}.png'.format(scene_idx, view_idx))
    # load image
    image = cv2.imread(image_path)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _, _
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    # save the annotated grounding dino image
    cv2.imwrite("EfficientSAM/LightHQSAM/groundingdino_annotated_image.jpg", annotated_frame)

    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy),
        torch.from_numpy(detections.confidence),
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=False,
                hq_token_only=True,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)


    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    pred_mask = np.zeros((image.shape[0], image.shape[1]))
    for inst_id, inst_mask in enumerate(detections.mask):
        pred_mask[inst_mask] = inst_id + 1

    pred_mask = (pred_mask / np.max(pred_mask)) * 255
    result = Image.fromarray(pred_mask.astype(np.uint8))
    mask_save_path = os.path.join(mask_save_root, 'scene_{:04}'.format(scene_idx), camera_type)
    os.makedirs(mask_save_path, exist_ok=True)
    result.save(os.path.join(mask_save_path, '{:04}.png'.format(view_idx)))

# # annotate image with detections
# box_annotator = sv.BoxAnnotator()
# mask_annotator = sv.MaskAnnotator()
# labels = [
#     f"{CLASSES[class_id]} {confidence:0.2f}"
#     for _, _, confidence, class_id, _, _
#     in detections]
# annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
# annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
#
# # save the annotated grounded-sam image
# cv2.imwrite("grounded_repvit_sam_annotated_image.jpg", annotated_image)
