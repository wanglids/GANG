from light_gaussian import _C
import torch
import torch.nn.functional as F
from utils.graphics_utils import getProjectionMatrix
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from copy import deepcopy

from light_gaussian import GaussianRasterizationSettings,GaussianRasterizer

def get_canonical_rays(H: int, W: int, tan_fovx: float, tan_fovy: float) -> torch.Tensor:
    cen_x = W / 2
    cen_y = H / 2
    focal_x = W / (2.0 * tan_fovx)
    focal_y = H / (2.0 * tan_fovy)

    x, y = torch.meshgrid(
        torch.arange(W),
        torch.arange(H),
        indexing="xy",
    )
    x = x.flatten()  # [H * W]
    y = y.flatten()  # [H * W]
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cen_x + 0.5) / focal_x,
                (y - cen_y + 0.5) / focal_y,
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [H * W, 3]
    # NOTE: it is not normalized
    return camera_dirs.cuda()


def getWorld2ViewTorch(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R[:3, :3].T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


# inverse the mapping from https://github.com/NVlabs/nvdiffrec/blob/dad3249af8ede96c7dd72c30328272117fabb710/render/light.py#L22
def get_envmap_dirs(res = [256, 512]) -> torch.Tensor:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0, 1.0 - 1.0 / res[0], res[0], device="cuda"),
        torch.linspace(-1.0, 1.0 - 1.0 / res[1], res[1], device="cuda"),
        indexing="ij",
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]

    return reflvec

def get_depth_cubemap(get_xyz,get_opacity,get_scaling,get_rotation,get_features, position, res = 512
):
    canonical_rays = get_canonical_rays(H=res, W=res, tan_fovx=1.0, tan_fovy=1.0)  # [HW, 3]
    norm = torch.norm(canonical_rays, p=2, dim=-1).reshape(res, res, 1)  # [H, W]

    bg_color = torch.zeros([3, res, res], device="cuda")
    rotations: List[torch.Tensor] = [
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, -1.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 0.0, -1.0]), torch.tensor([0.0, 1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
    ]
    zfar = 100.0
    znear = 0.01
    projection_matrix = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=np.pi * 0.5, fovY=np.pi * 0.5)
        .transpose(0, 1)
        .cuda()
    )

    depth_cubemap = []
    opacity_cubemap = []
    for r_idx, rotation in enumerate(rotations):
        c2w = rotation
        c2w[:3, 3] = position
        w2c = torch.inverse(c2w)
        T = w2c[:3, 3]
        R = w2c[:3, :3].T
        world_view_transform = getWorld2ViewTorch(R, T).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        input_args = (
            bg_color,
            # bg_colors[r_idx],
            get_xyz,
            torch.Tensor([]),
            get_opacity,
            get_scaling,
            get_rotation,
            torch.Tensor([]),
            get_features,
            camera_center,  # campos,
            world_view_transform,  # viewmatrix,
            full_proj_transform,  # projmatrix,
            1.0,  # scale_modifier
            1.0,  # tanfovx,
            1.0,  # tanfovy,
            res,  # image_height,
            res,  # image_width,
            1,
            False,  # prefiltered,
            True,  # argmax_depth, 
        )
        (num_rendered, rendered_image, opacity_map, radii, depth_map) = _C.lite_rasterize_gaussians(*input_args)

        # depth_cubemap.append(depth_map.permute(1, 2, 0) * norm)
        depth_cubemap.append(depth_map.permute(1, 2, 0))
        opacity_cubemap.append(opacity_map.permute(1, 2, 0))

    return torch.stack(depth_cubemap), torch.stack(opacity_cubemap)


# def get_depth_cubemap_moving(get_xyz,get_opacity,get_scaling,get_rotation,get_features,rotations, position, res = 256
# ):
#     # get canonical ray and its norm to normalize depth
#     canonical_rays = get_canonical_rays(H=res, W=res, tan_fovx=1.0, tan_fovy=1.0)  # [HW, 3]
#     norm = torch.norm(canonical_rays, p=2, dim=-1).reshape(res, res, 1)  # [H, W]

#     bg_color = torch.zeros([3, res, res], device="cuda")
    
#     zfar = 100.0
#     znear = 0.01
#     projection_matrix = (
#         getProjectionMatrix(znear=znear, zfar=zfar, fovX=np.pi * 0.5, fovY=np.pi * 0.5)
#         .transpose(0, 1)
#         .cuda()
#     )

#     depth_cubemap = []
#     opacity_cubemap = []
#     for r_idx, rotation in enumerate(rotations):
#         print(r_idx)
#          c2w = rotations[r_idx]
#         # print(c2w.shape,position.shape,type(c2w),type(position))
#         c2w[:3, 3] = position
#         w2c = torch.inverse(c2w)
#         T = w2c[:3, 3]
#         R = w2c[:3, :3].T
#         world_view_transform = getWorld2ViewTorch(R, T).transpose(0, 1)
#         full_proj_transform = (
#             world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
#         ).squeeze(0)
#         camera_center = world_view_transform.inverse()[3, :3]

#         input_args = (
#             bg_color,
#             # bg_colors[r_idx],
#             get_xyz,
#             torch.Tensor([]),
#             get_opacity,
#             get_scaling,
#             get_rotation,
#             torch.Tensor([]),
#             get_features,
#             camera_center,  # campos,
#             world_view_transform,  # viewmatrix,
#             full_proj_transform,  # projmatrix,
#             1.0,  # scale_modifier
#             1.0,  # tanfovx,
#             1.0,  # tanfovy,
#             res,  # image_height,
#             res,  # image_width,
#             1,
#             False,  # prefiltered,
#             True,  # argmax_depth, 
#         )
#         (num_rendered, rendered_image, opacity_map, radii, depth_map) = _C.lite_rasterize_gaussians(*input_args)

#         # depth_cubemap.append(depth_map.permute(1, 2, 0) * norm)
#         depth_cubemap.append(depth_map.permute(1, 2, 0))
#         opacity_cubemap.append(opacity_map.permute(1, 2, 0))

#     return torch.stack(depth_cubemap), torch.stack(opacity_cubemap)

    


def turbo_cmap(gray: np.ndarray) -> np.ndarray:
    """
    Visualize a single-channel image using matplotlib's turbo color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.turbo(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)



def DistributionGGX(
    normals: torch.Tensor,  # [H, W, 3]
    half_dirs: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
) -> torch.Tensor:
    a = roughness * roughness
    a2 = a * a
    NoH = saturate_dot(normals, half_dirs)
    
    NoH2 = NoH * NoH

    nom = a2
    denom = (NoH2 * (a2 - 1.0) + 1.0)
    denom = np.pi * denom * denom + 1e-4
    # print("nom",nom.max(),nom.min())
    # print("denom",denom.max(),denom.min())
    # print("NoH2",NoH2.max(),NoH2.min())

    return nom / denom

def saturate_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True).clamp(min=0.0, max=1.0)


def GeometrySchlickGGX(
    NoV: torch.Tensor, # [H, W, 1]
    roughness: torch.Tensor,  # [H, W, 1]
) -> torch.Tensor:
    r = roughness + 1.0
    k = (r * r) / 8.0
    nom = NoV
    denom = NoV * (1.0 - k) + k

    return nom / denom

def GeometrySmith(
    normals: torch.Tensor,  # [H, W, 3]
    view_dirs: torch.Tensor,  # [H, W, 3]
    light_dirs: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
) -> torch.Tensor:
    NoV = saturate_dot(normals, view_dirs)
    NoL = saturate_dot(normals, light_dirs)
    ggx2 = GeometrySchlickGGX(NoV, roughness)
    ggx1 = GeometrySchlickGGX(NoL, roughness)

    return ggx1 * ggx2


def fresnelSchlick(
    HoV: torch.Tensor,  # [H, W, 1]
    F0: torch.Tensor,  # [H, W, 3]
) -> torch.Tensor:
    return F0 + (1.0 - F0) * torch.pow((1.0 - HoV).clamp(0.0, 1.0), 5)

def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError


# https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/6.pbr/2.2.1.ibl_specular/2.2.1.pbr.fs
def light_pbr_shading(
    light_position: torch.Tensor,  # [3]
    light_intensity: torch.Tensor,  # [3]
    points: torch.Tensor,  # [H, W, 3]
    normals: torch.Tensor,  # [H, W, 3]
    view_dirs: torch.Tensor,  # [H, W, 3]
    albedo: torch.Tensor,  # [H, W, 3]
    roughness: torch.Tensor,  # [H, W, 1]
    mask: torch.Tensor,  # [H, W, 1]
    linear: bool = False,
    metallic: Optional[torch.Tensor] = None,
    shadow: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
) -> Dict:
    if background is None:
        background = torch.zeros_like(normals)  # [H, W, 3]

    # preapre
    light_dirs = F.normalize(light_position - points, p=2, dim=-1)  # [H, W, 3]
    half_dirs = (light_dirs + view_dirs) / 2.0  # [H, W, 3]
    distance = torch.norm(light_position - points, p=2, dim=-1, keepdim=True)  # [H, W, 1]
    attenuation = 1.0 / torch.pow(distance, 2)  # [H, W, 1]
    radiance = light_intensity * attenuation  # [H, W, 3]

    if metallic is None:
        F0 = torch.ones_like(albedo) * 0.04  # [H, W, 3]
    else:
        F0 = (1.0 - metallic) * 0.04 + albedo * metallic  # [H, W, 3]

    # Cook-Torrance BRDF
    NoV = saturate_dot(normals, view_dirs)  # [H, W, 1]
    NoL = saturate_dot(normals, light_dirs)  # [H, W, 1]
    HoV = saturate_dot(half_dirs, view_dirs)  # [H, W, 1]
    NDF = DistributionGGX(normals=normals, half_dirs=half_dirs, roughness=roughness)  # [H, W, 1]
    G = GeometrySmith(normals=normals, view_dirs=view_dirs, light_dirs=light_dirs, roughness=roughness)  # [H, W, 1]
    fresnel = fresnelSchlick(HoV=HoV, F0=F0)  # [H, W, 3]

    numerator = NDF * G * fresnel  # [H, W, 3]
    denominator = 4.0 * NoV * NoL + 1e-4  # [H, W, 1]
    specular = numerator / denominator + 1e-4  # [H, W, 3]

    kd = 1.0 - fresnel  # [H, W, 3]
    if metallic is not None:
        kd *= (1.0 - metallic)
    
    render_rgb = (kd * albedo / np.pi + specular) * radiance # * NoL

    render_rgb = torch.where(mask, render_rgb, background)

    if shadow is not None:
        render_rgb = torch.where(shadow == 0.0, render_rgb*0.2, render_rgb)

    # if linear:
    render_rgb = linear_to_srgb(render_rgb.squeeze())

    results = {}
    results.update(
        {
            "render_rgb": render_rgb,
        }
    )

    return results