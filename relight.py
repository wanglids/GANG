import os
from argparse import ArgumentParser
from os import makedirs
from typing import Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render, prefilter_voxel
# from pbr import CubemapLight, get_brdf_lut, pbr_shading
from scene.NVDIFFREC import Hybridlight,get_envmap_dirs,load_env

from scene import Scene
from utils.general_utils import safe_state
import nvdiffrast.torch as dr
import pyexr
from lpips import LPIPS
from utils.image_utils import psnr as get_psnr
from utils.loss_utils import ssim as get_ssim
import json
from PIL import Image

def read_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    if path.endswith(".hdr"):
        with open(path, "rb") as h:
            buffer_ = np.frombuffer(h.read(), np.uint8)
        rgb = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = pyexr.open(path).get()[:, :, :3]
    return rgb


def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map: torch.Tensor, res: List[int]) -> torch.Tensor:
    cubemap = torch.zeros(
        6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device="cuda"
    )
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(
            latlong_map[None, ...], texcoord[None, ...], filter_mode="linear"
        )[0]
    return cubemap


def render_set(model_path,name,light_name,scene,hdri,light,irradiance,occul_arg,pipeline,eval_env = False):
    iteration = scene.loaded_iter
    if name == "train":
        views = scene.getTrainCameras()
    elif name == "test":
        views = scene.getTestCameras()
    else:
        raise ValueError

    # build mip for environment light
    print(args.model_path)
    
    print("eval_env:",eval_env)
    relight_path = os.path.join(model_path, "relighting", light_name)
    makedirs(relight_path, exist_ok=True)

    relight_iamge_path = os.path.join(args.model_path, "relighting", light_name,"images")
    makedirs(relight_iamge_path, exist_ok=True)

    relight_pbr_env_path = os.path.join(args.model_path, "relighting", light_name,"pbr_env")
    makedirs(relight_pbr_env_path, exist_ok=True)

    render_albedo_path = os.path.join(args.model_path,"eval/albedo")

    light.build_mips()


    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    psnr_avg = 0.0
    ssim_avg = 0.0
    lpips_avg = 0.0
    bg = 1 
    lpips_fn = LPIPS(net="vgg").cuda()

    if eval_env:
        test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
        with open(test_transforms_file) as json_file:
            contents = json.load(json_file)
        frames = contents["frames"]

    gaussians = scene.gaussians
    t = 0
    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

            background[...] = 0.0  # NOTE: set zero
            
            gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)

            rendering_result = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, is_pbr=True,is_training=False,light=light)
            
            render_rgb = rendering_result['render'].clamp(min=0.0, max=1.0)
            render_albedo = rendering_result["albedo"]

            if not os.path.exists(os.path.join(render_albedo_path, f"{idx:05d}.png")):
                torchvision.utils.save_image(
                    render_albedo, os.path.join(render_albedo_path, f"{idx:05d}.png")
                )


            mask = view.mask


            if eval_env:
                
                if "Synthetic4Relight" in args.source_path:
                    image_path = os.path.join(args.source_path,
                                            "test_rli/" + light_name + "_" + frames[idx]["file_path"].split("/")[-1] + ".png")

                    g_image1 = imageio.imread(image_path)
                    g_image = g_image1[...,:3] / 255.0   
                    gt_image = torch.from_numpy(g_image).float().permute(2, 0, 1).to("cuda")

                else:
                    image_path = os.path.join(args.source_path, frames[idx]["file_path"]+ "_" + light_name + ".png")
                    g_image1 = Image.open(image_path)

                    g_image = torch.from_numpy(np.array(g_image1)[...,:3]) / 255.0

                    if len(g_image.shape) == 3:
                        gt_image = g_image.permute(2, 0, 1).to("cuda")
                    else:
                        gt_image = g_image.unsqueeze(dim=-1).permute(2, 0, 1).to("cuda")

                gt_image = gt_image * mask + bg * (1 - mask)
                render_rgb = render_rgb * mask + bg * (1 - mask)

                with torch.no_grad():
                    psnr_avg += get_psnr(gt_image, render_rgb).mean().double()
                    ssim_avg += get_ssim(gt_image, render_rgb).mean().double()
                    lpips_avg += lpips_fn(gt_image, render_rgb).mean().double()


            torchvision.utils.save_image(
                render_rgb, os.path.join(relight_iamge_path, f"{idx:05d}_{light_name}.png")
            )
            
    if eval_env:
        psnr = psnr_avg / len(views)
        ssim = ssim_avg / len(views)
        lpips = lpips_avg / len(views)
        print(f"psnr_pbr: {psnr}",f"ssim_pbr: {ssim}",f"lpips_pbr: {lpips}")
        print(os.path.join(args.model_path, "relighting", light_name, "metric.txt"))
        with open(os.path.join(args.model_path, "relighting", light_name, "metric.txt"), "w") as f:
            f.write(f"psnr_pbr: {psnr}\n")
            f.write(f"ssim_pbr: {ssim}\n")
            f.write(f"lpips_pbr: {lpips}\n")



def launch(
    model_path: str,
    checkpoint_str: str,
    # hdri_path: str,
    dataset: GroupParams,
    pipeline: GroupParams,
    skip_train: bool,
    skip_test: bool,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = False,
    is_pbr: bool=True
) -> None:


    # load hdri
    print("source_path:",args.source_path)
    eval_env = False

    if "TensoIRSynthetic" in args.source_path:
        task_names =  ['bridge', 'city', 'courtyard', 'fireplace', 'forest', 'night', 'snow', 'sunset'] # some envmap for relighting test
        task_dict = {}
        for task_name in task_names:
            task_dict[task_name] = {
                 "envmap_path": f"/home/dqli/program/data/env_maps/high_res_envmaps_1k/{task_name}.hdr",
            }
        eval_env = True
    else:
        task_names = ['bridge', 'city', 'courtyard', 'fireplace', 'forest', 'night', 'snow', 'sunset']
        task_dict = {}
        for task_name in task_names:
                task_dict[task_name] = {
                    "envmap_path": f"/home/dqli/program/data/env_maps/high_res_envmaps_1k/{task_name}.hdr",
                }


    irradiance_volumes = None

    occu_arg = None


    source_model_path = dataset.model_path
    source_source_path = dataset.source_path

    args.source_path = source_source_path
    args.model_path = source_model_path
    
    checkpoint_str =dataset.model_path+"/chkpnt40000.pth"
    
    gaussians = GaussianModel(
    dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
    dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
    dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend,is_pbr=is_pbr,
    normal_detal = dataset.normal_detal)

    checkpoint = torch.load(checkpoint_str)
    if isinstance(checkpoint, Tuple):
        model_params = checkpoint[0]
    elif isinstance(checkpoint, Dict):
        model_params = checkpoint["gaussians"]
    else:
        raise TypeError
    gaussians.restore(model_params)
    gaussians.eval()


    scene = Scene(dataset, gaussians, shuffle=False,load_iteration=-1)

    for task_name in task_names:
        hdri_path = task_dict[task_name]["envmap_path"]
        print(f"read hdri from {hdri_path}")
        hdri = read_hdr(hdri_path)
        hdri = torch.from_numpy(hdri).cuda()

        res = 256    
        numSG = 16
        cubemap = load_env(hdri_path,sg_path=None,res=res,numLgtSGs = numSG,is_sg = False)
        cubemap.eval()


        render_set(
            model_path=model_path,
            name="test",
            light_name=task_name,
            scene=scene,
            hdri=hdri,
            light=cubemap,
            irradiance = irradiance_volumes,
            occul_arg=occu_arg,
            pipeline=pipeline,
            eval_env = eval_env
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--hdri", type=str, default=None, help="The path to the hdri for relighting.")
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    parser.add_argument("--tone", action="store_true", help="Enable aces film tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear_to_sRGB for gamma correction.")
    parser.add_argument("--metallic", action="store_true", help="Enable metallic material reconstruction.")
    args = get_combined_args(parser)

    model_path = os.path.dirname(args.checkpoint)
    print("Rendering " + model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    launch(
        model_path=model_path,
        checkpoint_str=args.checkpoint,
        # hdri_path=args.hdri,
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        metallic=args.metallic,
        tone=args.tone,
        gamma=args.gamma,
    )


