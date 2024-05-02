import os
import pandas as pd

def match_name_fun(local_image_path, data):
    # 读取CSV文件
    df = data
    # 获取CSV文件中的文件名列表
    csv_filenames = df['file_name'].tolist()

    # 定义本地图片文件夹路径
    directory_path = local_image_path

    # 提取本地文件名和CSV中的编号映射
    csv_filename_mapping = {name.split('-')[0]: name for name in csv_filenames}

    # 提取本地文件编号映射
    local_filenames = os.listdir(directory_path)
    local_filename_mapping = {name.split('-')[0]: name for name in local_filenames}

    # 遍历本地文件，查找不匹配项并重命名
    for local_id, local_filename in local_filename_mapping.items():
        if local_id in csv_filename_mapping and local_filename != csv_filename_mapping[local_id]:
            old_path = os.path.join(directory_path, local_filename)
            new_name = csv_filename_mapping[local_id]
            new_path = os.path.join(directory_path, new_name)
            # 重命名文件
            os.rename(old_path, new_path)
            print(f'Renamed "{local_filename}" to "{new_name}"')