import argparse
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import glob
from scipy.spatial.transform import Rotation
import numpy as np
from tqdm import tqdm

from mvs_utils import export_nerfstudio, serialize_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qar_rec_folder", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)
    args = parser.parse_args()

    data_folder = args.qar_rec_folder
    assert data_folder

    res_img_folder = Path(args.out_dir).joinpath("images")
    if os.path.exists(res_img_folder):
        shutil.rmtree(res_img_folder)

    res_img_folder.mkdir(exist_ok=True, parents=True)
    res_cams = Path(args.out_dir).joinpath("cams_meta.npy")
    res_image_list = Path(args.out_dir).joinpath("image_list.txt")
    res_transforms = Path(args.out_dir).joinpath("transforms.json")

    image_paths = sorted(glob.glob(data_folder + "*.ppm"),key=lambda val: int(extract_image_id(val)))
    n_images = len(image_paths)
    image_list = []
    poses = []
    cams2pix = []
    bounds = []
    dist_params = []
    img_width = None
    img_height = None
    for image_path in tqdm(image_paths):
        index = extract_image_id(image_path)
        index_out =  f"{int(index):08d}"
        img = Image.open(image_path)
        res_img_path = res_img_folder.joinpath(index_out+".jpg")
        img.save(res_img_path)
        image_list.append(os.path.abspath(res_img_path))

        h, w = img.height, img.width
        img_width = w
        img_height = h
        pose_file = os.path.join(data_folder, "pose"+index+".txt")
        with open(pose_file, "r") as f:
            angles = [float(val) for val in f.readline().split(" ")]
            intrinsics = frustum_angles_to_intrinsics(
                angles[0], angles[1], angles[2], angles[3], w,h)

            bounds_line = f.readline().strip().split(" ")
            near = float(bounds_line[0])
            far = float(bounds_line[1])
            quaternion = np.array([float(val) for val in f.readline().strip().split(" ")])
            translation = np.array([float(val) for val in f.readline().strip().split(" ")])
            view2world = quatpose_to_mat(quaternion, translation)
            pose = view2world[:3, :4]

            dist_params.append(np.array([0,0,0,0]))
            poses.append(pose)
            bounds.append(np.array([near, far]))
            cams2pix.append(intrinsics)

    poses_np = np.stack(poses)
    cams2pix_np = np.stack(cams2pix)
    bounds_np = np.stack(bounds)
    dist_params_np = np.stack(dist_params)

    data = np.concatenate([poses_np.reshape([n_images, -1]),
                            cams2pix_np.reshape([n_images, -1]),
                            dist_params_np.reshape([n_images, -1]),
                            bounds_np.reshape([n_images, -1])], axis=-1)
    data = np.ascontiguousarray(data.astype(np.float64))
    np.save(res_cams, data)

    with open(res_image_list, "w") as f:
        f.writelines(image_list)

    frames = export_nerfstudio(
        image_list, poses, cams2pix, bounds, dist_params, img_width, img_height)
    serialize_frames(res_transforms, frames)


def proc(x):
    return np.ascontiguousarray(np.array(x).astype(np.float64))

def extract_image_id(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    index = filename[3:]
    return index

def quatpose_to_mat(quaternion, camera_position):
    # Convert the quaternion to a rotation matrix
    rotation = Rotation.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()

    # Create a 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = camera_position

    return transform_matrix



def frustum_angles_to_intrinsics(
    angle_left, angle_right, angle_up, angle_down, image_width, image_height):
    # Calculate the horizontal FOV angle
    fov_horizontal_rad = abs(angle_right - angle_left)

    # Calculate the vertical FOV angle
    fov_vertical_rad = abs(angle_down - angle_up)

    # Calculate the focal lengths
    focal_length_horizontal = image_width / (2 * np.tan(fov_horizontal_rad / 2))
    focal_length_vertical = image_height / (2 * np.tan(fov_vertical_rad / 2))

    # Calculate the principal point
    principal_point_x = image_width / 2
    principal_point_y = image_height / 2

    # Construct the intrinsic matrix
    intrinsics_matrix = np.array([[focal_length_horizontal, 0, principal_point_x],
                                  [0, focal_length_vertical, principal_point_y],
                                  [0, 0, 1]])

    return intrinsics_matrix


if __name__ == '__main__':
    main()
