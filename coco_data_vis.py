import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.mask import decode
import numpy as np
import random


def random_color():
    return [random.random() for _ in range(3)]  # 生成随机颜色

def visualize_coco_data(json_file):
    # 获取JSON文件所在目录的路径
    json_dir = os.path.dirname(json_file)
    
    # 加载JSON数据
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 初始化COCO对象
    coco = COCO(json_file)

    # 获取类别信息
    category_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

    # 处理每张图像和其标注
    for img_info in data['images']:
        image_path = os.path.join(json_dir, '../images',img_info['file_name'])  # 构建图像的完整路径
        image = Image.open(image_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        ax = plt.gca()

        # 获取该图像的所有标注
        ann_ids = coco.getAnnIds(imgIds=[img_info['id']])
        annotations = coco.loadAnns(ann_ids)

        # 在图像上绘制每一个标注
        for ann_id, ann in enumerate(annotations):
            # 获取颜色和类别名
            color = random_color()
            category_name = category_id_to_name[ann['category_id']]

            # 绘制边界框
            if 'bbox' in ann:
                bbox = ann['bbox']
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                plt.text(bbox[0], bbox[1] - 10, f"{ann_id}", color=color, fontsize=12, weight='bold')

            # 绘制分割
            if 'segmentation' in ann:
                # 检查分割数据类型并解码RLE
                if isinstance(ann['segmentation'], dict):  # RLE格式
                    mask = decode(ann['segmentation'])
                    masked_image = np.ma.masked_where(mask == 0, mask)
                    plt.imshow(masked_image, cmap='cool', alpha=0.5, interpolation='none')

        plt.axis('off')
        # 保存图像
        output_path = os.path.join(json_dir, 'output', f"{img_info['file_name']}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

# 调用函数，使用示例
visualize_coco_data('/media/user/4TB-1/dataset/syntable_coco_sample/annotations/train.json')