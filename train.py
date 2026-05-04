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
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
# torch.autograd.detect_anomaly()
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
import lpips
from random import randint
from utils.loss_utils import l1_loss,render_normal_from_depth, ssim, get_tv_loss, get_masked_tv_loss,predicted_normal_loss,delta_normal_loss

from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state,clusteranchor
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


# from Baking import recon_occlusion, IrradianceVolumes
from scene.NVDIFFREC import Hybridlight,get_envmap_dirs

from torchvision.utils import save_image, make_grid
import nvdiffrast.torch as dr
from torch.nn import functional as F
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

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    is_pbr = dataset.is_pbr

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
        dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend,is_pbr,
        dataset.normal_detal,dataset.with_meta
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales,is_pbr=is_pbr)
    gaussians.training_setup(opt)
    gaussians.set_coarse_interval(opt.coarse_iter, opt.coarse_factor)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if dataset.random_background:
        bg_color = [np.random.random(), np.random.random(), np.random.random()]
    elif dataset.white_background:
        bg_color = [1.0, 1.0, 1.0]
    else:
        bg_color = [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if is_pbr:
        anchor = gaussians.get_anchor
        sg_position = clusteranchor(anchor,K=16)
        envmap_dirs = get_envmap_dirs()
        cubemap = Hybridlight(base_res=256, num_sg = 16, inital_position = sg_position) 
        cubemap.training_setup(opt)
    else:
        cubemap = None



    pbr_kwargs = dict()
    
    for iteration in range(first_iter, opt.iterations + 1):        

        # network gui not available in octree-gs yet
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if is_pbr:
            cubemap.update_learning_rate(iteration)

        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        gaussians.set_anchor_mask(viewpoint_cam.camera_center, iteration, viewpoint_cam.resolution_scale)
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)

        
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, is_pbr=is_pbr,light=cubemap, retain_grad=retain_grad,iteration=iteration) 
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]
        depth,normal = render_pkg['depth'],render_pkg['precomput_normal']

        alpha = render_pkg["alpha"].detach()[0]
        image = torch.clamp(image, 0.0, 1.0)
        gt_image = viewpoint_cam.original_image

        Ll1 = l1_loss(image, gt_image)
        ssim_loss = (1.0 - ssim(image, gt_image))
        image_psnr = psnr(image, gt_image).mean().double()
        
        loss_dict = {}

        loss_dict["image_loss"] = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss 
 

        
        if iteration >= 10000 and (iteration - 10000) % 2500 <= 500 - 1 and not is_pbr: 
            anchor = gaussians.get_anchor[voxel_visible_mask]
            anchor_p = gaussians.position_normal(anchor)

            view_anchor = anchor - viewpoint_cam.camera_center.repeat(anchor.shape[0], 1)
            view_anchor_normal = view_anchor/view_anchor.norm(dim=1, keepdim=True) # (N, 3)
            scaling_anchor = gaussians.get_scaling[:,3:][voxel_visible_mask]
            rot_anchor = gaussians.get_rotation[voxel_visible_mask]
            anchor_normal = gaussians.computeNorm(scaling_anchor, rot_anchor, view_anchor_normal)

            psr_anchor = gaussians.dpsr(anchor_p.unsqueeze(0),anchor_normal.unsqueeze(0))
            psr_anchor = torch.tanh(psr_anchor)

            position_neural = gaussians.position_normal(render_pkg["points"])

            psr_point = gaussians.dpsr(position_neural.unsqueeze(0),render_pkg["points_normal"].unsqueeze(0))
            psr_point = torch.tanh(psr_point)

            loss_dict["psr_loss"] = 0.01*F.mse_loss(psr_anchor, psr_point)
        
        if gaussians.normal_detal:
            if iteration >5000:
                normal_from_depth = render_normal_from_depth(viewpoint_cam, depth) * alpha
                surface_mask = alpha > opt.omit_opacity_threshold # H, W
                loss_dict['normal_loss'] = 0.01*predicted_normal_loss(normal, normal_from_depth, surface_mask, threshold=opt.omit_opacity_threshold)

                loss_dict['delta_reg'] = opt.pseudo*delta_normal_loss(render_pkg["delta_normal"], render_pkg["alpha"])
        
        else:
            if iteration >5000:
                normal_from_depth = render_normal_from_depth(viewpoint_cam, depth) * alpha
                surface_mask = alpha > opt.omit_opacity_threshold # H, W
                loss_dict['normal_loss'] = 0.01*predicted_normal_loss(normal, normal_from_depth, surface_mask, threshold=opt.omit_opacity_threshold)
                
        if iteration>3000:
            loss_dict["local_loss"] = opt.lambda_local * render_pkg["local_loss"]

        if is_pbr:
            normal_mask = (render_pkg["precomput_normal"] != 0).all(0).unsqueeze(0)
            if gaussians.with_matallic:
                all_meta = torch.cat([render_pkg["albedo"], render_pkg["roughness"], render_pkg["matallic"]], dim=0)
            else:
                all_meta = torch.cat([render_pkg["albedo"], render_pkg["roughness"]], dim=0)
            if (normal_mask == 0).sum() > 0:
                brdf_tv_loss = get_masked_tv_loss( 
                    normal_mask,
                    gt_image,  # [3, H, W]
                    all_meta,  # [5, H, W]
                )
            else:
                brdf_tv_loss = get_tv_loss(
                    gt_image,  
                    all_meta,  
                    pad=1,  
                    step=1,
                )

            loss_dict["meta_loss"] = opt.lambda_brdf_tv * brdf_tv_loss #+ lamb_loss * lamb_weight

            envmap = dr.texture(
                cubemap.base[None, ...],
                envmap_dirs[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )[
                0
            ]  # [H, W, 3]
            tv_h1 = torch.pow(envmap[1:, :, :] - envmap[:-1, :, :], 2).mean()
            tv_w1 = torch.pow(envmap[:, 1:, :] - envmap[:, :-1, :], 2).mean()
            env_tv_loss = tv_h1 + tv_w1
            loss_dict["env_tv_loss"] = env_tv_loss * 0.01


        loss = sum( v for k, v in loss_dict.items())
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                if is_pbr:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}",
                                              "Img_PSNR": f"{image_psnr:.{3}f}"},)
                else:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                              "Img_PSNR": f"{image_psnr:.{3}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), is_pbr,cubemap, wandb, logger)
            if (iteration in saving_iterations) or iteration==opt.iterations:
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if is_pbr:
                    cubemap.save_light(scene.model_path + "/Hybridlight" + str(iteration) + ".npy")

            # densification
            if iteration < opt.update_until and iteration > opt.start_stat and is_pbr is False:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if opt.update_anchor and iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(
                        iteration=iteration,
                        check_interval=opt.update_interval, 
                        success_threshold=opt.success_threshold,
                        grad_threshold=opt.densify_grad_threshold, 
                        update_ratio=dataset.update_ratio,
                        extra_ratio=dataset.extra_ratio,
                        extra_up=dataset.extra_up,
                        min_opacity=opt.min_opacity
                    )
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if is_pbr:
                    cubemap.optimizer.step()
                    cubemap.optimizer.zero_grad(set_to_none=True)
                    # irradiance_optimizer.step()
                    # irradiance_optimizer.zero_grad(set_to_none=True)
                    cubemap.clamp_(min=0.0,max=1.0)

                for component in pbr_kwargs.values():
                    try:
                        component.step()
                    except:
                        pass
            
            if (iteration in checkpoint_iterations) or iteration==opt.iterations:
                for com_name, component in pbr_kwargs.items():
                    try:
                        torch.save((component.capture(), iteration),
                                   os.path.join(scene.model_path, f"{com_name}_chkpnt" + str(iteration) + ".pth"))
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    except:
                        pass

                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))
            
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                if is_pbr:
                    cubemap.save_light(scene.model_path + "/Hybridlight" + str(iteration) + ".npy")
            
    if dataset.eval:
        eval_render(scene, gaussians, pipe, background, opt.iterations, pbr_kwargs,is_pbr,light=cubemap)



def eval_render(scene, gaussians, pipe, background, iteration, pbr_kwargs,is_pbr=False,light=None):
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    gaussians.get_color_mlp.training = False
    test_cameras = scene.getTestCameras()
    os.makedirs(os.path.join(args.model_path, 'eval', 'render'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, 'eval', 'depth'), exist_ok=True)

    if is_pbr:
        os.makedirs(os.path.join(args.model_path, 'eval', 'albedo'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'roughness'), exist_ok=True)
        os.makedirs(os.path.join(args.model_path, 'eval', 'matallic'), exist_ok=True)


    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        t = 0
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            gaussians.set_anchor_mask(viewpoint.camera_center, iteration, viewpoint.resolution_scale)
            voxel_visible_mask = prefilter_voxel(viewpoint, gaussians, pipe, background)

            results = render(viewpoint, gaussians, pipe, background, visible_mask=voxel_visible_mask, is_pbr=is_pbr,light=light,is_training=False)

            image =  torch.clamp(results["render"],0.0,1.0)
            gt_image = torch.clamp(viewpoint.original_image,0.0,1.0)

            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += lpips_fn(image, gt_image).mean().double()

            save_image(image, os.path.join(args.model_path, 'eval', "render", f"{viewpoint.image_name}_{t}.png"))
            save_image(gt_image, os.path.join(args.model_path, 'eval', "gt", f"{viewpoint.image_name}_{t}.png"))

            normal = results["precomput_normal"]
            save_image(normal*0.5+0.5, os.path.join(args.model_path, 'eval', "normal", f"{viewpoint.image_name}_{t}.png"))

            if is_pbr:
                albedo = results["albedo"]
                save_image(albedo, os.path.join(args.model_path, 'eval', "albedo", f"{viewpoint.image_name}_{t}.png"))
                roughness = results["roughness"]
                save_image(roughness, os.path.join(args.model_path, 'eval', "roughness", f"{viewpoint.image_name}_{t}.png"))


            render_depth = results['depth']
            render_depth = apply_depth_colormap(-render_depth[0][...,None]).permute(2,0,1)
        
            torchvision.utils.save_image(
                render_depth,os.path.join(args.model_path, 'eval', "depth",f"{idx:05d}.png")
            )
            t += 1 

    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    lpips_test /= len(test_cameras)
    with open(os.path.join(args.model_path, 'eval', f"eval_{args.iterations}.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")
    print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                       lpips_test))





def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,is_pbr= False, light=None,wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss,"iter_time":elapsed})
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                  {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    scene.gaussians.set_anchor_mask(viewpoint.camera_center, iteration, viewpoint.resolution_scale)
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs,visible_mask=voxel_visible_mask, is_pbr=is_pbr,light=light)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        visible_count_list.append(visible_count)

        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
        
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim, 
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level, 
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, resolution_scales=dataset.resolution_scales)
        gaussians.eval()

        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()] 
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, eval_name, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / eval_name

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[3000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # parser.add_argument("-env_c", "--checkpoint", type=str, default=None)
    parser.add_argument("--gpu", type=str, default = '-1')
    args = parser.parse_args(sys.argv[1:])

    # enable logging
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    logger.info(f'args: {args}')

    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(10000, args.iterations + 1, 10000)]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != args.iterations:
        args.test_iterations.append(args.iterations)
    print(args.test_iterations)

    # if args.save_iterations[0] == -1:
    #     args.save_iterations = [i for i in range(10000, args.iterations + 1, 10000)]
    # if len(args.save_iterations) == 0 or args.save_iterations[-1] != args.iterations:
    #     args.save_iterations.append(args.iterations)

    args.save_iterations = [args.iterations]

    print(args.save_iterations)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Octree-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # # All done
    logger.info("\nTraining complete.")

    # rendering
    # logger.info(f'\nStarting Rendering~')
    # if args.eval:
    #     visible_count = render_sets(lp.extract(args), -1, pp.extract(args), skip_train=True, skip_test=False, wandb=wandb, logger=logger)
    # else:
    #     visible_count = render_sets(lp.extract(args), -1, pp.extract(args), skip_train=False, skip_test=True, wandb=wandb, logger=logger)
    # logger.info("\nRendering complete.")

    # calc metrics
    # logger.info("\n Starting evaluation...")
    # eval_name = 'test' if args.eval else 'train'
    # evaluate(args.model_path, eval_name, visible_count=visible_count, wandb=wandb, logger=logger)
    # logger.info("\nEvaluating complete.")



