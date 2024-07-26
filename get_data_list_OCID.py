import os
import cv2
import numpy as np

def valid_images_with_masks(root_path):
    valid_images = []

    # 遍历root_path下的所有文件夹（假设这是数据集的顶层文件夹）
    for dataset_folder in os.listdir(root_path):
        dataset_path = os.path.join(root_path, dataset_folder)
        if os.path.isdir(dataset_path):
            # 遍历dataset_path下的“floor”和“table”文件夹
            for place_folder in ["floor", "table"]:
                # if place_folder == 'table':
                #     ignore_first = True
                # else:
                #     ignore_first = False
                ignore_first = False
                place_path = os.path.join(dataset_path, place_folder)
                if os.path.isdir(place_path):
                    # 遍历“bottom”和“top”文件夹
                    for position_folder in ["bottom", "top"]:
                        position_path = os.path.join(place_path, position_folder)
                        if os.path.isdir(position_path):
                            # 遍历所有“seqxx”或物体名文件夹
                            for category_folder in os.listdir(position_path):
                                category_path = os.path.join(position_path, category_folder)
                                if os.path.isdir(category_path):
                                    if category_folder.startswith('seq'):
                                        # 检查“rgb”和“label”文件夹中的文件
                                        valid_images += check_images_and_masks(category_path, ignore_first)
                                    else:
                                        # 如果是物体名文件夹，继续查找“seqxx”文件夹
                                        for subfolder in os.listdir(category_path):
                                            if subfolder.startswith('seq'):
                                                sub_folder_path = os.path.join(category_path, subfolder)
                                                valid_images += check_images_and_masks(sub_folder_path, ignore_first)

    return valid_images

def check_images_and_masks(folder_path, ignore_first=False):
    rgb_path = os.path.join(folder_path, 'rgb')
    label_path = os.path.join(folder_path, 'label')
    valid_images_list = []

    if os.path.isdir(rgb_path) and os.path.isdir(label_path):
        # 获取所有图片文件，并按文件名排序以确保时间顺序
        image_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')])

        # 对于每一个图片，检查是否有对应的mask，并且mask中是否有物体
        for image_file in image_files:
            image_path = os.path.join(rgb_path, image_file)
            mask_path = os.path.join(label_path, image_file)  # 假设mask文件和图片文件名相同
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                print(rgb_path, np.unique(mask))
                if mask is not None and np.any(mask > 0):  # 检查mask是否有物体
                    relative_image_path = os.path.relpath(image_path, start=data_list_save_root)
                    valid_images_list.append(relative_image_path)

    return valid_images_list


# 替换此路径为你的数据集根目录
dataset_root_path = '/media/user/4TB-1/dataset/UOIS/OCID'
data_list_save_root = '/media/user/4TB-1/dataset/UOIS/OCID'
valid_images = valid_images_with_masks(dataset_root_path)
print(f"Total valid images: {len(valid_images)}")

with open(os.path.join(data_list_save_root, "data_list.txt"), 'w') as f:
    for i in valid_images:
        f.write(i+'\n')