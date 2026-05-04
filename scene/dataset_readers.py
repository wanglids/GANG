#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import glob
import sys
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
import imageio as imageio
import cv2
from scene.gaussian_model import BasicPointCloud

try:
    import laspy
except:
    print("No laspy")



from scene.utils import load_img_rgb, load_mask_bool, load_depth, load_pfm

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    normal: np.array
    albedo: np.array
    roughness: np.array
    metal: np.array
    irradiance: np.array
    image_mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)


        meta_path = os.path.join(os.path.dirname(images_folder),"RGB_X", f"{image_name}.npy")
        if os.path.exists(meta_path):
            meta_all = np.load(meta_path, allow_pickle=True)
            albedo = meta_all.item()['albedo']
            roughness = meta_all.item()["roughness"]
            metallic = meta_all.item()["metallic"] 
            irradiance = meta_all.item()["irradiance"]
        else:
            albedo = roughness = metallic = irradiance = normal = None

        image_mask = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,normal=normal,albedo=albedo,roughness=roughness,metal=metallic,irradiance=irradiance,image_mask=image_mask,image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)

    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchLas(path):
    las = laspy.read(path)
    positions = np.vstack((las.x, las.y, las.z)).transpose()
    try:
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.random.rand(positions.shape[0], positions.shape[1])

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def read_las_file(path):
    las = laspy.read(path)
    positions = np.vstack((las.x, las.y, las.z)).transpose()
    try:
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.random.rand(positions.shape[0], positions.shape[1])

    return positions, colors, normals

def read_multiple_las_files(paths, ply_path):
    all_positions = []
    all_colors = []
    all_normals = []

    for path in paths:
        positions, colors, normals = read_las_file(path)
        all_positions.append(positions)
        all_colors.append(colors)
        all_normals.append(normals)

    all_positions = np.vstack(all_positions)
    all_colors = np.vstack(all_colors)
    all_normals = np.vstack(all_normals)

    print("Saving point cloud to .ply file...")
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(all_positions.shape[0], dtype=dtype)
    attributes = np.concatenate((all_positions, all_normals, all_colors), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

    return BasicPointCloud(points=all_positions, colors=all_colors, normals=all_normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, ds, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if ds == 1 else f"images_{ds}"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    # try:
    print(f'start fetching data from ply file')
    pcd = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms3(path, transformsfile, white_background, extension=".png", debug=False):
    cam_infos = []
    is_train = ("train" in transformsfile)

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            image_path = os.path.join(path, frame["file_path"] + extension)
            mask_path = image_path.replace("_rgb.exr", "_mask.png")
            image_name = Path(image_path).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image = load_img_rgb(image_path).astype(np.float32)
            mask = load_mask_bool(mask_path).astype(np.float32)
            # mask = mask[:,:,np.newaxis]

            fovy = focal2fov(fov2focal(fovx, image.shape[0]), image.shape[1])
            
            meta_path = os.path.join(path, "RGB_X", f"{image_name}.npy")
            if is_train and os.path.exists(meta_path):
                meta_all = np.load(meta_path, allow_pickle=True)
                albedo = meta_all.item()['albedo']
                normal = meta_all.item()["normal"]
                roughness = meta_all.item()["roughness"]
                metallic = meta_all.item()["metallic"] 
                irradiance = meta_all.item()["irradiance"]
            else:
                normal=albedo=roughness=metallic=irradiance=None

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image,normal=normal,albedo=albedo,roughness=roughness,metal=metallic,irradiance=irradiance, image_mask=mask,
                                        image_path=image_path, image_name=image_name,
                                        width=image.shape[1], height=image.shape[0]))

            if debug and idx >= 5:
                break

    return cam_infos


def readSynthetic4RelightInfo(path, white_background, eval, debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms3(path, "transforms_train.json", white_background, "_rgb.exr", debug=debug)
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms3(path, "transforms_test.json", white_background, "_rgba.png", debug=debug)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        normals = np.random.randn(*xyz.shape)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        storePly(ply_path, xyz, SH2RGB(shs) * 255, normals)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info


def readCamerasFromTransforms4(path, transformsfile, white_background, extension=".png", debug=False):
    cam_infos = []
    is_train = ("train" in transformsfile)
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            image_path = os.path.join(path, frame["file_path"] + extension)
            # mask_path = image_path.replace("_rgb.exr", "_mask.png")
            image_name = Path(image_path).stem

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            bg = 1 if white_background else 0


            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            
            bg = [0.0, 0.0, 0.0]
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            mask = norm_data[:, :, 3:4][:,:,0]
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            meta_path = os.path.join(path, "RGB_X", f"{image_name}.npy")
            if is_train and os.path.exists(meta_path):
                meta_all = np.load(meta_path, allow_pickle=True)
                albedo = meta_all.item()['albedo']
                normal = meta_all.item()["normal"]
                roughness = meta_all.item()["roughness"]
                metallic = meta_all.item()["metallic"] 
                irradiance = meta_all.item()["irradiance"]
            else:
                normal=albedo=roughness=metallic=irradiance=None

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fovy, FovX=fovx, image=image,normal=normal,albedo=albedo,roughness=roughness,metal=metallic,irradiance=irradiance, image_mask=mask,
                                        image_path=image_path, image_name=image_name,
                                        width=image.size[0], height=image.size[1]))

            if debug and idx >= 5:
                break

    return cam_infos


def readTensorIRSyntheticInfo(path, white_background, eval, debug=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms4(path, "transforms_train.json", white_background, ".png", debug=debug)
    if eval:
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms4(path, "transforms_test.json", white_background, ".png", debug=debug)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        normals = np.random.randn(*xyz.shape)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        storePly(ply_path, xyz, SH2RGB(shs) * 255, normals)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Synthetic4Relight": readSynthetic4RelightInfo,
    "TensoIRSynthetic": readTensorIRSyntheticInfo,
}