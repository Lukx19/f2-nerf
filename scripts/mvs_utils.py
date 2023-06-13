import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def format_index(index: int) -> str:
    return f"{index:08d}"

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ").reshape(
        (4, 4)
    )
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(
        " ".join(lines[7:10]), dtype=np.float32, sep=" "
    ).reshape((3, 3))

    params_line = lines[11].split()
    depth_min = float(params_line[0])
    depth_max = float(params_line[-1])
    return intrinsics, extrinsics, depth_min, depth_max

def read_pair_file(filename, min_support_views=5) -> Dict[int, List[int]]:
    data = {}
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) >= min_support_views:
                data[ref_view] = src_views
    return data

def find_camera_file(dataset_folder:Path)-> Optional[Path]:
    cameras_folders = []
    cameras_folders.append(dataset_folder.joinpath("cams"))
    cameras_folders.append(dataset_folder.joinpath("cams_0"))
    cameras_folders.append(dataset_folder.joinpath("cams_1"))
    cameras_folders.append(dataset_folder.joinpath("cams_2"))

    for cam_path in cameras_folders:
        # print(cam_path)
        if os.path.exists(cam_path) is True:
            return cam_path
    return None

def find_images_folder(dataset_folder:Path)->Optional[Path]:
    images_folders = []
    images_folders.append(dataset_folder.joinpath("images"))
    images_folders.append(dataset_folder.joinpath("blended_images"))


    for img_path in images_folders:
        # print(cam_path)
        if os.path.exists(img_path) is True:
            return img_path
    return None


def find_pair_file(dataset_folder: Path) -> Optional[Path]:
    root_pair_file = dataset_folder.joinpath("pair.txt")

    cameras_folder = find_camera_file(dataset_folder)
    if cameras_folder is None:
        return None
    cams_pair_file = cameras_folder.joinpath("pair.txt")

    valid_path = cams_pair_file
    for path in [root_pair_file, cams_pair_file]:
        if os.path.exists(path) is True:
            valid_path = path
            break

    return valid_path