import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import numpy as np
from functools import singledispatch

from mvs_utils import find_camera_file, find_images_folder, find_pair_file, format_index, read_camera_parameters, read_pair_file

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
    res_transforms = Path(args.out_dir).joinpath("transforms.json")

    pairs = read_pair_file(pair_file, 1)
    keys = list(sorted(pairs.keys()))
    end_index = args.end_index
    if end_index == -1:
        end_index = len(keys)
    keys = keys[args.start_index: end_index]
    n_images = len(keys)
    frames = FramesData(0,0,0,0,0,0,0,0,[])
    for i, key in enumerate(keys):
        filename = format_index(key) + ".jpg"
        img_path = Path(img_folder).joinpath(filename)
        res_img_path = Path(res_img_folder).joinpath(filename).absolute()
        shutil.copyfile(img_path, res_img_path)

        intrinsics, extrinsics, depth_min, depth_max = read_camera_parameters(cam_folder.joinpath(format_index(key) + "_cam.txt"))
        # Convert extrinsics to camera-to-world.
        pose = np.linalg.inv(extrinsics)
        # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
        pose = pose @ np.diag([1, -1, -1, 1])

        frame = Frame("images/"+filename, pose.astype(np.float64).tolist(),
                      intrinsics[0,0], intrinsics[1,1],
                      intrinsics[0,2], intrinsics[1,2])
        frames.frames.append(frame)

    with open(res_transforms, "w") as f:
        frames_dict = asdict(frames)
        data_str = json.dumps(frames_dict, default=to_serializable)
        f.write(data_str)

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)

@dataclass
class Frame:
    file_path: str
    # Frame coordinate system: +X is right, +Y is up, and +Z is pointing back and away from the camera.
    transform_matrix: List[List[float]] # view2world matrix
    fl_x: int # focal length x
    fl_y: int # focal length y
    cx: int # principal point x
    cy: int # principal point y

@dataclass
class FramesData:
#   fl_x: int # focal length x
#   fl_y: int # focal length y
#   cx: int # principal point x
#   cy: int # principal point y
  w: int #image width
  h: int #image height
  k1: int # first radial distorial parameter, used by [OPENCV, OPENCV_FISHEYE]
  k2: int # second radial distorial parameter, used by [OPENCV, OPENCV_FISHEYE]
  k3: int # third radial distorial parameter, used by [OPENCV_FISHEYE]
  k4: int # fourth radial distorial parameter, used by [OPENCV_FISHEYE]
  p1: int # first tangential distortion parameter, used by [OPENCV]
  p2: int # second tangential distortion parameter, used by [OPENCV]
  frames: List[Frame] #r-frame intrinsics and extrinsics parameters
  camera_model: str = "OPENCV" # camera model type [OPENCV, OPENCV_FISHEYE]

if __name__ == '__main__':
    main()
