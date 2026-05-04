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
from os import makedirs
import torch
import numpy as np

from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
import lpips
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
from typing import Dict, List, Tuple
from scene.NVDIFFREC import Hybridlight
import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from torchvision.utils import save_image
from PIL import Image


from utils.graphics_utils import getWorld2View2, getProjectionMatrix



from matplotlib import cm
def apply_colormap(image, cmap="viridis"):
    colormap = cm.get_cmap(cmap)
    colormap = torch.tensor(colormap.colors).to(image.device)  # type: ignore
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return colormap[image_long[..., 0]]

def apply_depth_colormap(depth, cmap="turbo", min=None, max=None):
    near_plane = float(torch.min(depth)) if min is None else min
    far_plane = float(torch.max(depth)) if max is None else max

    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0, 1)

    colored_image = apply_colormap(depth, cmap=cmap)
    return colored_image

def render_set(model_path,pipe, name, cubemap,  iteration, views, gaussians, pipeline, background, show_level,  is_pbr=False):

    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    print("save paht : ",args.model_path,iteration)
    os.makedirs(os.path.join(args.model_path, 'eval_new', 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval_new', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval_new', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval_new', 'depth'), exist_ok=True)

    
    if show_level:
        render_level_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_level")
        makedirs(render_level_path, exist_ok=True)

    t = 0
    cubemap.build_mips()
    
    for idx, viewpoint in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize(); t0 = time.time()

        gaussians.set_anchor_mask(viewpoint.camera_center, iteration, viewpoint.resolution_scale)
        voxel_visible_mask = prefilter_voxel(viewpoint, gaussians, pipeline, background)
        results = render(viewpoint, gaussians, pipe, background, visible_mask=voxel_visible_mask, is_pbr=is_pbr,light=cubemap, is_training=False)

        image = results["render"]
        gt_image = viewpoint.original_image

        psnr_test += psnr(image, gt_image).mean().double()
        ssim_test += ssim(image, gt_image).mean().double()
        lpips_test += lpips_fn(image, gt_image).mean().double()
        t += 1
        
    psnr_test /= len(views)
    ssim_test /= len(views)
    lpips_test /= len(views)
    with open(os.path.join(args.model_path, 'eval', f"eval_{iteration}_new.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(iteration, "test", psnr_test, ssim_test,
                                                                       lpips_test))
    


def eval_brdf(data_root: str, scene: Scene, model_path: str, name: str) -> None:
    # only for TensoIR synthetic
    if name == "train":
        transform_file = os.path.join(data_root, "transforms_train.json")
    elif name == "test":
        transform_file = os.path.join(data_root, "transforms_test.json")

    with open(transform_file, "r") as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]

    iteration = scene.loaded_iter
    pbr_dir = os.path.join(model_path, name, f"ours_{iteration}", "pbr")

    albedo_psnr_avg = 0.0
    albedo_ssim_avg = 0.0
    albedo_lpips_avg = 0.0
 
    pbr_path = os.path.join(model_path, name, f"ours_{iteration}", "pbr")
    albedo_gts = []
    albedo_maps = []
    masks = []
    gt_albedo_list = []
    reconstructed_albedo_list = []

    lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

    for idx, frame in enumerate(tqdm(frames)):

        albedo_path = frame["file_path"].replace("rgba", "albedo") + ".png"
        albedo_gt = np.array(Image.open(os.path.join(data_root, albedo_path)))[..., :3]
        mask = np.array(Image.open(os.path.join(data_root, albedo_path)))[..., 3] > 0
        albedo_gt = torch.from_numpy(albedo_gt).cuda() / 255.0  # [H, W, 3]
        albedo_gts.append(albedo_gt)
        mask = torch.from_numpy(mask).cuda()  # [H, W]
        masks.append(mask)
        gt_albedo_list.append(albedo_gt[mask])
        # read prediction
        brdf_map = np.array(Image.open(os.path.join(pbr_dir, f"{idx:05}_brdf.png")))
        H, W3, _ = brdf_map.shape
        albedo_map = brdf_map[:, : (W3 // 3), :]  # [H, W, 3]
        albedo_map = torch.from_numpy(albedo_map).cuda() / 255.0  # [H, W, 3]
        albedo_maps.append(albedo_map)
        reconstructed_albedo_list.append(albedo_map[mask])
    gt_albedo_all = torch.cat(gt_albedo_list, dim=0)
    albedo_map_all = torch.cat(reconstructed_albedo_list, dim=0)
    # single_channel_ratio = (gt_albedo_all / albedo_map_all.clamp(min=1e-6))[..., 0].median()  # [1]
    three_channel_ratio, _ = (gt_albedo_all / albedo_map_all.clamp(min=1e-6)).median(dim=0)  # [3]

    for idx, (mask, albedo_map, albedo_gt) in enumerate(tqdm(zip(masks, albedo_maps, albedo_gts))):
        albedo_map[mask] *= three_channel_ratio
        albedo_map = albedo_map.permute(2, 0, 1)  # [3, H, W]
        albedo_gt = albedo_gt.permute(2, 0, 1)  # [3, H, W]
        torchvision.utils.save_image(albedo_map, os.path.join(pbr_path, f"{idx:05d}_albedo.png"))
        albedo_psnr_avg += psnr(albedo_gt, albedo_map).mean().double()
        albedo_ssim_avg += ssim(albedo_gt, albedo_map).mean().double()
        albedo_lpips_avg += lpips_fn(albedo_gt, albedo_map).mean().double()

    albedo_psnr = albedo_psnr_avg / (len(frames))
    albedo_ssim = albedo_ssim_avg / (len(frames))
    albedo_lpips = albedo_lpips_avg / (len(frames))
    print(f"albedo psnr_avg: {albedo_psnr}; ssim_avg: {albedo_ssim}; lpips_avg: {albedo_lpips}")

    
def render_sets(dataset : ModelParams,checkpoint_str, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, show_level : bool, ape_code : int, is_pbr : bool):
    with torch.no_grad():
        res = 256
        cubemap = Hybridlight(base_res=res).cuda()
        cubemap.load_light(dataset.model_path+ f"/Hybridlight40000.npy")
        env_map = cubemap.compute_env_envmap(return_img = True)

        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend, is_pbr=is_pbr,normal_detal = dataset.normal_detal)
        checkpoint = torch.load(checkpoint_str)
        if isinstance(checkpoint, Tuple):
            model_params = checkpoint[0]
        elif isinstance(checkpoint, Dict):
            model_params = checkpoint["gaussians"]
        else:
            raise TypeError
        gaussians.restore(model_params)
        # iteration = 30000
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
        iteration = scene.loaded_iter
        print("Load iteration: ",iteration)
        gaussians.eval()
        gaussians.plot_levels()
        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()] 
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        if is_pbr:
            res = 256
            cubemap = Hybridlight(base_res=res).cuda()

            cubemap.load_light(dataset.model_path+ f"/Hybridlight{iteration}.npy")
        else:
            cubemap = None

        # if not skip_train:
        #     render_set(dataset.model_path,pipeline, "train", cubemap, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, show_level,is_pbr)
        if not skip_test:
            render_set(dataset.model_path,pipeline, "test", cubemap, scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, show_level,is_pbr)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=10, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show_level", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    args.is_pbr = True

    render_sets(model.extract(args),args.checkpoint,args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.show_level, args.ape, args.is_pbr)
    
