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

from typing import NamedTuple
import torch.nn as nn
import torch
import kornia
from . import _C

class _surface_align(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, rotation, knn_index):
        loss_d, loss_normal, binning_buffer, mean_d = _C.surface_align(xyz, rotation, knn_index)

        # Keep relevant tensors for backward
        ctx.save_for_backward(xyz, rotation, binning_buffer, knn_index, mean_d)
        return loss_d, loss_normal

    @staticmethod
    def backward(ctx, grad_out_loss_d, grad_out_loss_normal):
        # print("backward :")
        # print("grad_out_loss_d :", grad_out_loss_d)
        # print("grad_out_loss_normal :", grad_out_loss_normal)
        # Restore necessary values from context
        xyz, rotation, binning_buffer, knn_index, mean_d = ctx.saved_tensors

        grad_xyz, grad_rotation = _C.surface_align_backward(xyz, rotation, binning_buffer, mean_d, knn_index, grad_out_loss_d, grad_out_loss_normal)

        grads = (
            grad_xyz,
            None,
            grad_rotation,
            None
        )

        return grads


class SurfaceAlign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz, rotation, knn_index):
        return _surface_align.apply(xyz, rotation, knn_index)

        
def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    norm3Ds_precomp,
    extra_attrs,
    raster_settings,
):
    num_contrib, color, depth, opacity, norm, alpha, radii, extra = _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        norm3Ds_precomp,
        extra_attrs,
        raster_settings,
    )
    
    norm = torch.nn.functional.normalize(norm, p=2, dim=0)

    focal_x = raster_settings.image_width / (2.0 * raster_settings.tanfovx)
    focal_y = raster_settings.image_height / (2.0 * raster_settings.tanfovy)
    # # NOTE: trick to smooth depth for better normal
    # depth_filter = depth
    depth_filter = kornia.filters.median_blur(depth[None, ...], (3, 3))[0]
    normal_from_depth = _C.depth_to_normal(
        raster_settings.image_width,
        raster_settings.image_height,
        focal_x,
        focal_y,
        raster_settings.viewmatrix,
        depth_filter,
    )
    
    # 3, H, W
    return num_contrib, color, depth, opacity, norm,normal_from_depth, alpha, radii, extra

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        norm3Ds_precomp,
        extra_attrs,
        raster_settings,
    ):
        # restrict the length of extra attr values to avoid dynamically sized shared memory allocation
        assert extra_attrs.shape[0] == 0 or extra_attrs.shape[1] <= 34
        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            norm3Ds_precomp,
            extra_attrs,
            extra_attrs.shape[1] if extra_attrs.shape[0] != 0 else 0,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, num_contrib, color, depth, opacity, norm, alpha, radii, extra, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_contrib, color, depth, opacity, norm, alpha, radii, extra, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, norm3Ds_precomp, radii, extra_attrs, sh, geomBuffer, binningBuffer, imgBuffer, alpha)
        return num_contrib, color, depth, opacity, norm, alpha, radii, extra

    @staticmethod
    def backward(ctx, grad_out_contrib, grad_out_color, grad_out_depth, grad_out_opacity, grad_out_norm, grad_out_alpha, _, grad_out_extra):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, norm3Ds_precomp, radii, extra_attrs, sh, geomBuffer, binningBuffer, imgBuffer, alpha = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                extra_attrs,
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                norm3Ds_precomp,
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_depth,
                grad_out_norm,
                grad_out_alpha,
                grad_out_extra,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                alpha,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_norm3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_extra_attrs = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_norm3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_extra_attrs = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_norm3Ds_precomp,
            grad_extra_attrs,
            None
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3Ds_precomp = None, norm3Ds_precomp=None, extra_attrs=None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3Ds_precomp is None) or ((scales is not None or rotations is not None) and cov3Ds_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            raise ValueError('To support norm and depth prediction, scales == None is not allowed')
            scales = torch.Tensor([])
        if rotations is None:
            raise ValueError('To support norm and depth prediction, rotations == None is not allowed')
            rotations = torch.Tensor([])
        if cov3Ds_precomp is None:
            cov3Ds_precomp = torch.Tensor([])
        if norm3Ds_precomp is None:
            norm3Ds_precomp = torch.Tensor([])
        if extra_attrs is None:
            extra_attrs = torch.Tensor([])
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3Ds_precomp,
            norm3Ds_precomp,
            extra_attrs,
            raster_settings, 
        )

    def visible_filter(self, means3D, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        with torch.no_grad():
            radii = _C.rasterize_gaussians_filter(means3D,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3D_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            raster_settings.prefiltered,
            raster_settings.debug)
        return  radii