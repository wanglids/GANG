import os
import glob
from argparse import ArgumentParser

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
from lpips import LPIPS
import torch

def get_mae(gt_normal_stack: np.ndarray, render_normal_stack: np.ndarray) -> float:
    # compute mean angular error
    MAE = np.mean(
        np.arccos(np.clip(np.sum(gt_normal_stack * render_normal_stack, axis=-1), -1, 1))
        * 180
        / np.pi
    )
    return MAE.item()


def get_ratio(output_dir,gt_dir):

    test_dirs = glob.glob(os.path.join(gt_dir, "test_*"))
    test_dirs.sort()
    
    lpips_fn = LPIPS(net="vgg").cuda()
    normal_gt_stack = []
    normal_gs_stack = []
    normal_from_depth_stack = []

    albedo_psnr_avg = 0.0
    albedo_ssim_avg = 0.0
    albedo_lpips_avg = 0.0
    albedo_gts = []
    albedo_maps = []
    masks = []
    gt_albedo_list = []
    reconstructed_albedo_list = []
    for test_dir in tqdm(test_dirs):
        if "test_179" in test_dir:
            continue
        test_id = int(test_dir.split("_")[-1])
        albedo_path = os.path.join(test_dir, "albedo.png")
        albedo_gt = np.array(Image.open(os.path.join(albedo_path)))[..., :3]
        mask = np.array(Image.open(os.path.join(albedo_path)))[..., 3] > 0
        albedo_gt = torch.from_numpy(albedo_gt).cuda() / 255.0  # [H, W, 3]
        albedo_gts.append(albedo_gt)
        mask = torch.from_numpy(mask).cuda()  # [H, W]
        masks.append(mask)
        gt_albedo_list.append(albedo_gt[mask])

        # gs normal
        albedo_map = np.array(Image.open(os.path.join(output_dir, "eval/albedo", f"rgba_{test_id}.png")))
        albedo_map = torch.from_numpy(albedo_map).cuda() / 255.0  # [H, W, 3]
        albedo_maps.append(albedo_map)
        reconstructed_albedo_list.append(albedo_map[mask])
    gt_albedo_all = torch.cat(gt_albedo_list, dim=0)
    albedo_map_all = torch.cat(reconstructed_albedo_list, dim=0)
    three_channel_ratio, _ = (gt_albedo_all / albedo_map_all.clamp(min=1e-6)).median(dim=0)  # [3]
    
    return np.array(three_channel_ratio)