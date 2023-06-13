import argparse
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import numpy as np

from scripts.mvs_utils import find_camera_file, find_images_folder, find_pair_file, format_index, read_camera_parameters, read_pair_file

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


if __name__ == '__main__':
    main()
