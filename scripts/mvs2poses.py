import argparse
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mvs_dataset", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    args = parser.parse_args()

    img_folder = find_images_folder(Path(args.mvs_dataset))
    pair_file = find_pair_file(Path(args.mvs_dataset))
    cam_folder = find_camera_file(Path(args.mvs_dataset))
    assert img_folder
    assert pair_file
    assert cam_folder

    res_img_folder = Path(args.out_dir).joinpath("images")
    res_img_folder.mkdir(exist_ok=True, parents=True)
    res_cams = Path(args.out_dir).joinpath("cams_meta.npy")
    res_image_list = Path(args.out_dir).joinpath("image_list.txt")

    image_list = []
    poses = []
    cams2pix = []
    bounds = []
    dist_params = []
    pairs = read_pair_file(pair_file, 1)
    keys = list(sorted(pairs.keys()))
    end_index = args.end_index
    if end_index == -1:
        end_index = len(keys)
    keys = keys[args.start_index: end_index]
    n_images = len(keys)
    for key in keys:
        filename = format_index(key) + ".jpg"
        img_path = Path(img_folder).joinpath(filename)
        res_img_path = Path(res_img_folder).joinpath(filename).absolute()
        image_list.append(str(res_img_path)+ "\n")
        shutil.copyfile(img_path, res_img_path)

        intrinsics, extrinsics, depth_min, depth_max = read_camera_parameters(cam_folder.joinpath(format_index(key) + "_cam.txt"))
        # Convert extrinsics to camera-to-world.
        c2w_mats = np.linalg.inv(extrinsics)
        pose = c2w_mats[:3, :4]
        # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
        pose = pose @ np.diag([1, -1, -1, 1])
        dist_params.append(np.array([0,0,0,0]))
        poses.append(pose)

        cams2pix.append(intrinsics)
        bounds.append(np.array([depth_min, depth_max]))


    poses = np.stack(poses)
    cams2pix = np.stack(cams2pix)
    bounds = np.stack(bounds)
    dist_params = np.stack(dist_params)

    data = np.concatenate([poses.reshape([n_images, -1]),
                            cams2pix.reshape([n_images, -1]),
                            dist_params.reshape([n_images, -1]),
                            bounds.reshape([n_images, -1])], axis=-1)
    data = np.ascontiguousarray(data.astype(np.float64))
    np.save(res_cams, data)

    with open(res_image_list, "w") as f:
        f.writelines(image_list)


def proc(x):
    return np.ascontiguousarray(np.array(x).astype(np.float64))

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

if __name__ == '__main__':
    main()
