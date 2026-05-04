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

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.graphics_utils import fov2focal
import numpy as np
import kornia
from utils.image_utils import erode
from kornia.filters import spatial_gradient

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)




def get_tv_loss(gt_image,prediction,pad = 1,step = 1):
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(
                -(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            rgb_grad_w = torch.exp(
                -(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)  # [C, H-1, W]
            tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2)  # [C, H, W-1]
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss


def depth2normal(depth, mask, camera):
    # conver to camera position
    camD = depth.permute([1, 2, 0])
    mask = mask.permute([1, 2, 0])
    shape = camD.shape
    device = camD.device
    h, w, _ = torch.meshgrid(torch.arange(0, shape[0]), torch.arange(0, shape[1]), torch.arange(0, shape[2]),
                             indexing='ij')
    # print(h)
    h = h.to(torch.float32).to(device)
    w = w.to(torch.float32).to(device)
    p = torch.cat([w, h], axis=-1)

    p[..., 0:1] -= 0.5 * camera.image_width
    p[..., 1:2] -= 0.5 * camera.image_height
    p *= camD
    K00 = fov2focal(camera.FoVy, camera.image_height)
    K11 = fov2focal(camera.FoVx, camera.image_width)
    K = torch.tensor([K00, 0, 0, K11]).reshape([2, 2])
    Kinv = torch.inverse(K).to(device)
    # print(p.shape, Kinv.shape)
    p = p @ Kinv.t()
    camPos = torch.cat([p, camD], -1)

    # padded = mod.contour_padding(camPos.contiguous(), mask.contiguous(), torch.zeros_like(camPos), filter_size // 2)
    # camPos = camPos + padded
    p = torch.nn.functional.pad(camPos[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    mask = torch.nn.functional.pad(mask[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)

    p_c = (p[:, 1:-1, 1:-1, :]) * mask[:, 1:-1, 1:-1, :]
    p_u = (p[:, :-2, 1:-1, :] - p_c) * mask[:, :-2, 1:-1, :]
    p_l = (p[:, 1:-1, :-2, :] - p_c) * mask[:, 1:-1, :-2, :]
    p_b = (p[:, 2:, 1:-1, :] - p_c) * mask[:, 2:, 1:-1, :]
    p_r = (p[:, 1:-1, 2:, :] - p_c) * mask[:, 1:-1, 2:, :]

    n_ul = torch.cross(p_u, p_l)
    n_ur = torch.cross(p_r, p_u)
    n_br = torch.cross(p_b, p_r)
    n_bl = torch.cross(p_l, p_b)

    # n_ul = torch.nn.functional.normalize(torch.cross(p_u, p_l), dim=-1)
    # n_ur = torch.nn.functional.normalize(torch.cross(p_r, p_u), dim=-1)
    # n_br = torch.nn.functional.normalize(torch.cross(p_b, p_r), dim=-1)
    # n_bl = torch.nn.functional.normalize(torch.cross(p_l, p_b), dim=-1)

    # n_ul = torch.nn.functional.normalize(torch.cross(p_l, p_u), dim=-1)
    # n_ur = torch.nn.functional.normalize(torch.cross(p_u, p_r), dim=-1)
    # n_br = torch.nn.functional.normalize(torch.cross(p_r, p_b), dim=-1)
    # n_bl = torch.nn.functional.normalize(torch.cross(p_b, p_l), dim=-1)

    n = n_ul + n_ur + n_br + n_bl
    n = n[0]

    # n *= -torch.sum(camVDir * camN, -1, True).sign() # no cull back

    mask = mask[0, 1:-1, 1:-1, :]

    # n = gaussian_blur(n, filter_size, 1) * mask

    n = torch.nn.functional.normalize(n, dim=-1)
    # n[..., 1] *= -1
    # n *= -1

    n = (n * mask).permute([2, 0, 1])
    return n

def cos_loss(output, gt, thrsh=0, weight=1):
    cos = torch.sum(output * gt * weight, 0)
    return (1 - cos[cos < np.cos(thrsh)]).mean()

def normal2curv(normal, mask):
    # normal = normal.detach()
    n = normal.permute([1, 2, 0])
    m = mask.permute([1, 2, 0])
    n = torch.nn.functional.pad(n[None], [0, 0, 1, 1, 1, 1], mode='replicate')
    m = torch.nn.functional.pad(m[None].to(torch.float32), [0, 0, 1, 1, 1, 1], mode='replicate').to(torch.bool)
    n_c = (n[:, 1:-1, 1:-1, :]      ) * m[:, 1:-1, 1:-1, :]
    n_u = (n[:,  :-2, 1:-1, :] - n_c) * m[:,  :-2, 1:-1, :]
    n_l = (n[:, 1:-1,  :-2, :] - n_c) * m[:, 1:-1,  :-2, :]
    n_b = (n[:, 2:  , 1:-1, :] - n_c) * m[:, 2:  , 1:-1, :]
    n_r = (n[:, 1:-1, 2:  , :] - n_c) * m[:, 1:-1, 2:  , :]
    curv = (n_u + n_l + n_b + n_r)[0]
    curv = curv.permute([2, 0, 1]) * mask
    curv = curv.norm(1, 0, True)
    return curv


def normal_loss(render_normal,render_depth,render_opacity,viewpoint_cam,opt):
    mask_vis = (render_opacity.detach() > 1e-5)
    # gt_normal = viewpoint_cam.normal
    render_normal = torch.nn.functional.normalize(render_normal, dim=-1)  * mask_vis
    # gt_normal = torch.nn.functional.normalize(gt_normal, dim=-1)  * mask_vis
    dep2nor = depth2normal(render_depth, mask_vis, viewpoint_cam)

    loss_pseudo = cos_loss(render_normal, dep2nor, thrsh=np.pi * 1 / 10000, weight=1)
    curv_n = normal2curv(render_normal, mask_vis)
    loss_curv = l1_loss(curv_n * 1, 0)

    # loss_N = cos_loss(render_normal, gt_normal)
    # loss_N = l2_loss(render_normal, gt_normal)

    loss_normal = opt.pseudo*loss_pseudo + opt.curv*loss_curv
    # loss_normal = opt.normal*loss_N

    return loss_normal




def get_masked_tv_loss(
    mask: torch.Tensor,  # [1, H, W]
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    erosion: bool = False,
) -> torch.Tensor:
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]

    # erode mask
    mask = mask.float()
    if erosion:
        kernel = mask.new_ones([7, 7])
        mask = kornia.morphology.erosion(mask[None, ...], kernel)[0]
    mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
    mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]
    # mask_h = mask[1:, :] * mask[:-1, :]  # [1, H-1, W]
    # mask_w = mask[:, 1:] * mask[:, :-1]  # [1, H, W-1]


    tv_loss = (tv_h * rgb_grad_h * mask_h).mean() + (tv_w * rgb_grad_w * mask_w).mean()

    return tv_loss


def predicted_normal_loss(normal, normal_ref, alpha=None, threshold=0.05):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: ( H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()
        weight[weight < threshold] = 0.0
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device)
    else:
        weight = torch.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    n = normal_ref.permute(1,2,0).reshape(-1,3)
    n_pred = normal.permute(1,2,0).reshape(-1,3)
    loss = (w * (1.0 - torch.sum(n * n_pred, axis=-1))).mean()

    return loss



def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss


def delta_normal_loss(delta_normal_norm, alpha=None):
    # delta_normal_norm: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(delta_normal_norm)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    l = delta_normal_norm.permute(1,2,0).reshape(-1,3)[...,0]
    loss = (w * l).mean()

    return loss

def first_order_edge_aware_loss(data, img):
    return (spatial_gradient(data[None], order=1)[0].abs() * torch.exp(-spatial_gradient(img[None], order=1)[0].abs())).sum(1).mean()


def eikonal_loss(sdf_gradients):
    gradient_error = (torch.linalg.norm(sdf_gradients.reshape(-1, 3), ord=2, dim=-1) - 1.0) ** 2
    return gradient_error.mean()


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz


def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth_pcd2normal(xyz):
    hd, wd,_ = xyz.shape 
    bottom_point = xyz[..., 2:hd,   1:wd-1, :]
    top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
    right_point  = xyz[..., 1:hd-1, 2:wd,   :]
    left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    _, xyz_cam = depth2point_cam(depth[None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3).reshape(*depth.shape, 3)[0]

    xyz_normal = depth_pcd2normal(xyz_cam)

    return xyz_normal


def render_normal_from_depth(viewpoint_cam, depth):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()
    normal_ref = normal_from_depth_image(depth, 
                                        intrinsic_matrix.to(depth.device), 
                                        extrinsic_matrix.to(depth.device))

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref
