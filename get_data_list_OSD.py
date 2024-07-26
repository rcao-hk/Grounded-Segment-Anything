import os
import re

def natural_sort_key(s):
    """ 自然排序支持函数，用于提取字符串中的数字进行排序 """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def get_png_files_with_relative_paths(search_directory, save_file_path):
    """ 
    获取指定目录及子目录下所有的.png文件的相对路径，并将它们保存到指定的文本文件中。
    保存的相对路径以search_directory的最后一个目录名开始。
    """
    base_dir = os.path.basename(search_directory)
    png_files = []
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            if file.endswith('.png'):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, start=os.path.dirname(search_directory))
                png_files.append(relative_path)

    png_files.sort(key=natural_sort_key)

    with open(os.path.join(save_file_path, 'data_list.txt'), 'w') as file:
        for path in png_files:
            file.write(path + '\n')

# 你的具体路径
search_directory_path = '/media/user/4TB-1/dataset/UOIS/OSD/image_color'
save_file_path = '/media/user/4TB-1/dataset/UOIS/OSD'
get_png_files_with_relative_paths(search_directory_path, save_file_path)

