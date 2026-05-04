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
from einops import repeat

import math
# from depth_normal_gauss import GaussianRasterizationSettings,GaussianRasterizer
# from light_geo_gauss import GaussianRasterizationSettings,GaussianRasterizer,SurfaceAlign
from light_gaussian import GaussianRasterizationSettings,GaussianRasterizer,SurfaceAlign
from scene.gaussian_model import GaussianModel
import numpy as np
import torch.nn.functional as F
from utils.sh_utils import eval_sh
from utils.graphics_utils import rgb_to_srgb
# from Baking import recon_occlusion
# import open3d as o3d
from utils.graphics_utils import normal_from_depth_image
from utils.loss_utils import eikonal_loss

def debug_hook(module, input, output):
    if torch.isnan(output).any():
        print(f"NaN detected in {module.__class__.__name__}")
        print("Input range:", input[0].min(), input[0].max())
        print("Output range:", output.min(), output.max())
        raise ValueError("NaN encountered")
    

def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def local_var(inputs):
    # input N M C
    aa = torch.var(inputs,dim=1)

    return torch.mean(torch.sum(aa,dim=-1))

def local_var_normal(inputs,mask):
    # input N M C
    unique_classes = mask.unique()
    variances = []
    for cls in unique_classes:

        cls_mask = (mask == cls).squeeze()
        cls_data = inputs[cls_mask]

        if cls_data.size(0) > 1:  
            cls_variance = cls_data.var(dim=1, unbiased=False)
            variances.append(cls_variance)
    if variances:
        variances = torch.cat(variances, dim=0)  
        overall_mean_variance = variances.mean()  

    return overall_mean_variance



def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, iteration= 0, ape_code=-1, is_pbr=False):
    ## view frustum filtering for acceleration    
    global roughness, albedo, matallic
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc.get_anchor_feat[visible_mask]
    level = pc.get_level[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    sdf_loss = 0

    local_loss = 0
    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        if pc.add_level:
            cat_view = torch.cat([ob_view, level], dim=1)
        else:
            cat_view = ob_view
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    if pc.add_level:
        cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1+1]
        cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1) # [N, c+3+1]
    else:
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        if ape_code < 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            appearance = pc.get_appearance(camera_indicies)
        else:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
            appearance = pc.get_appearance(camera_indicies)


    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    
    if pc.dist2level=="progressive":
        prog = pc._prog_ratio[visible_mask]
        transition_mask = pc.transition_mask[visible_mask]
        prog[~transition_mask] = 1.0
        neural_opacity = neural_opacity * prog

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)


    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)


    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]
    

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]

    grid_rotation = pc._rotation[visible_mask]

    concatenated = torch.cat([grid_scaling, grid_rotation,anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)

    if pc.normal_detal and iteration>500:
        flag = 1
    elif iteration>3000:
        flag =1
    else:
        flag=0

    if is_training and flag ==1:
    
        scaling_repeat_all, rotation_repeat_all, repeat_anchor_all = concatenated_repeated.split([6, 4,3], dim=-1)
        offsets_all = offsets * scaling_repeat_all[:,:3]
        xyz_all = repeat_anchor_all + offsets_all
        rot_all = pc.rotation_activation(scale_rot[:,3:7])
        index = torch.linspace(0,xyz_all.shape[0]-1,xyz_all.shape[0]).int().cuda()
        index = index.reshape([anchor.shape[0],pc.n_offsets])
 
        pair_d_loss, pair_normal_loss = SurfaceAlign()(xyz_all,rot_all,index)  
        local_loss +=  0.05*torch.mean(pair_d_loss) + 0.01*torch.mean(pair_normal_loss)


    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat,rotation_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6,4, 3, 3, 7, 3], dim=-1)
    

    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])

    rot = pc.rotation_activation(rotation_repeat*scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets    

    view_dir = xyz - viewpoint_camera.camera_center.repeat(xyz.shape[0], 1)
    view_dir_normal = (view_dir/view_dir.norm(dim=1, keepdim=True)).detach() # (N, 3)

    if pc.normal_detal:
        if pc.add_opacity_dist:
            delta_normal1 = pc.get_normal1_mlp(cat_local_view)  # [N, k]
            delta_normal2 = pc.get_normal2_mlp(cat_local_view)
        else:
            delta_normal1 = pc.get_normal1_mlp(cat_local_view_wodist)
            delta_normal2 = pc.get_normal2_mlp(cat_local_view_wodist)
        delta_normal1 =delta_normal1.reshape([anchor.shape[0]*pc.n_offsets, 3])
        delta_normal2 =delta_normal2.reshape([anchor.shape[0]*pc.n_offsets, 3])
        normal,delta_normal = pc.computeNorm(scaling, rot,view_dir_normal, delta_normal1,delta_normal2)
        delta_normal_norm = delta_normal.norm(dim=1, keepdim=True)*0.1
    else:
        normal = pc.computeNorm(scaling, rot,view_dir_normal)
        delta_normal_norm = None


    if is_pbr:
        matallic = None
        if pc.add_opacity_dist:
            roughness = pc.get_roughness_mlp(cat_local_view)  # [N, k]
            albedo = pc.get_albedo_mlp(cat_local_view)
            if pc.with_matallic:
                matallic = pc.get_matallic_mlp(cat_local_view)
        else:
            roughness = pc.get_roughness_mlp(cat_local_view_wodist)
            albedo = pc.get_albedo_mlp(cat_local_view_wodist)
            if pc.with_matallic:
                matallic = pc.get_matallic_mlp(cat_local_view_wodist)

        if is_training:
        
            albedo_loss = local_var(albedo.reshape([anchor.shape[0],pc.n_offsets, 3]))
            roughness_loss = local_var(roughness.reshape([anchor.shape[0],pc.n_offsets, 1]))
            if pc.with_matallic:
                metrics_loss = local_var(matallic.reshape([anchor.shape[0],pc.n_offsets, 1]))
            if pc.with_matallic:
                local_loss += albedo_loss+roughness_loss+metrics_loss
            else:
                local_loss += albedo_loss+roughness_loss
    
        albedo = albedo.reshape([anchor.shape[0]*pc.n_offsets, 3])
        roughness = roughness.reshape([-1, 1])
        if pc.with_matallic:
            matallic = matallic.reshape([-1,1])

        if pc.with_matallic:
            concatenated_all = torch.cat([albedo, roughness, matallic], dim=-1)
            masked = concatenated_all[mask]
            albedo, roughness, matallic = masked.split([3, 1, 1], dim=-1)
        else:
            concatenated_all = torch.cat([albedo, roughness], dim=-1)
            masked = concatenated_all[mask]
            albedo, roughness = masked.split([3, 1], dim=-1)

        albedo =  torch.clamp(albedo, 0.0, 1.0)
        roughness =  torch.clamp(roughness, 0.001, 1.0)
        matallic =  torch.clamp(matallic, 0.0, 1.0)

    else:
        albedo = None
        roughness = None
        matallic = None



    return xyz, color, opacity, scaling, rot, neural_opacity, mask, albedo, roughness, matallic,normal,delta_normal_norm,local_loss,sdf_loss
   


def scale_loss(scaling):

    sorted_scale, _ = torch.sort(scaling, dim=-1)
    min_scale_loss = sorted_scale[...,0]
    loss_scale = 100.0*min_scale_loss.mean()

    return loss_scale


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None,is_pbr=False,light=None, retain_grad=False, is_training =True, Local_pkg=None,iteration = 0,ape_code=-1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # if is_training:
    xyz, color, opacity, scaling, rot, neural_opacity, mask, albedo, roughness, matallic,normal,delta_normal_norm,local_loss,sdf_loss = generate_neural_gaussians(viewpoint_camera, pc,visible_mask,is_training=is_training,is_pbr=is_pbr,iteration= iteration,ape_code = ape_code)

    loss_scale = scale_loss(scaling)
    if pc.normal_detal:
        delta_normal_norm = delta_normal_norm.repeat(1, 3)


    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)



    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if is_pbr:

        viewdirs = F.normalize(viewpoint_camera.camera_center - xyz, p=2, dim=-1)

        light.build_mips()
        
        normal_t = normal * 0.5 + 0.5 

        light_color, extras = light.lightRender(xyz, normal_t, albedo, roughness, matallic, viewdirs)

        if is_training:
            normal = normal @ viewpoint_camera.world_view_transform[:3, :3]
        normal = normal * 0.5 + 0.5

        if pc.with_matallic:
            if pc.normal_detal:
                features = torch.cat([normal,delta_normal_norm,albedo,roughness,matallic],dim=-1)             
            else:
                features = torch.cat([normal,albedo,roughness,matallic],dim=-1)
        else:
            if pc.normal_detal:
                features = torch.cat([normal,delta_normal_norm,albedo,roughness],dim=-1)             
            else:
                features = torch.cat([normal,albedo,roughness],dim=-1)              

        color = light_color
    else:
        if is_training:           
            normal = normal @ viewpoint_camera.world_view_transform[:3, :3]
        normal = normal * 0.5 + 0.5

        if pc.normal_detal:
            features = torch.cat([normal,delta_normal_norm],dim=-1)
        else:
            features = normal


    n_contri,rendered_image, rendered_depth,rendered_opacity, rendered_norm,depth_normal, rendered_alpha, radii, rendered_features = rasterizer(
        means3D=xyz,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=color,
        opacities=opacity,
        scales=scaling,
        rotations=rot,
        cov3Ds_precomp=None,
        extra_attrs=features
    )
    feature_dict = {}

    if is_pbr:
        if pc.with_matallic:
            if pc.normal_detal:
                precomput_normal,delta_normal_t,rendered_albedo,rendered_roughness,rendered_matallic = rendered_features.split([3,3,3,1,1], dim=0)
            else:
                precomput_normal,rendered_albedo,rendered_roughness,rendered_matallic = rendered_features.split([3,3,1,1], dim=0)
                delta_normal_t = None

            feature_dict.update({"albedo": rendered_albedo,
                            "roughness": rendered_roughness,
                            "matallic": rendered_matallic
                            })
        else:
            if pc.normal_detal:
                precomput_normal,delta_normal_t,rendered_albedo,rendered_roughness = rendered_features.split([3,3,3,1], dim=0)             
            else:
                precomput_normal,rendered_albedo,rendered_roughness = rendered_features.split([3,3,1], dim=0)
                delta_normal_t = None
            feature_dict.update({"albedo": rendered_albedo,
                            "roughness": rendered_roughness
                            })             
    else:
        if pc.normal_detal:
            precomput_normal,delta_normal_t = rendered_features.split([3, 3],dim=0)
        else:
            precomput_normal = rendered_features
            delta_normal_t = None
            

    precomput_normal = (precomput_normal- 0.5) * 2
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # if is_training:
    if is_pbr:
        results = {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "neural_opacity": neural_opacity,
            "selection_mask": mask,
            "scaling": scaling,
            "normal": rendered_norm,
            "precomput_normal": precomput_normal,
            "delta_normal":delta_normal_t,
            "depth_normal":depth_normal,
            "depth": rendered_depth,
            "opacity": rendered_opacity,
            "alpha": rendered_alpha,
            "local_loss": local_loss,
            "scale_loss":loss_scale,
            "sdf_loss":sdf_loss,
            "points":xyz,
            "points_normal":normal
            }
        results.update(feature_dict)

        return results        
        
    else:
        return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling": scaling,
            "normal": rendered_norm,
            "precomput_normal": precomput_normal,
            "delta_normal":delta_normal_t,
            "depth_normal":depth_normal,
            "depth": rendered_depth,
            "opacity": rendered_opacity,
            "alpha": rendered_alpha,
            "local_loss": local_loss,
            "scale_loss":loss_scale,
            "sdf_loss":sdf_loss,
            "points":xyz,
            "points_normal":normal

            }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,anchor_mask=None, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)


    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    if anchor_mask is None:
        means3D = pc.get_anchor[pc._anchor_mask]
    else:
        means3D = pc.get_anchor[anchor_mask]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:

        if anchor_mask is None:
            scales = pc.get_scaling[pc._anchor_mask]
            rotations = pc.get_rotation[pc._anchor_mask]
        else:
            scales = pc.get_scaling[anchor_mask]
            rotations = pc.get_rotation[anchor_mask]

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii_pure > 0

    return visible_mask
