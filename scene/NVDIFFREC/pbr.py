import os
from typing import Dict, Optional, Union

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from .light import EnvironmentLight

def envBRDF_approx(roughness: torch.Tensor, NoV: torch.Tensor) -> torch.Tensor:
    c0 = torch.tensor([-1.0, -0.0275, -0.572, 0.022], device=roughness.device)
    c1 = torch.tensor([1.0, 0.0425, 1.04, -0.04], device=roughness.device)
    c2 = torch.tensor([-1.04, 1.04], device=roughness.device)
    r = roughness * c0 + c1
    a004 = (
        torch.minimum(torch.pow(r[..., (0,)], 2), torch.exp2(-9.28 * NoV)) * r[..., (0,)]
        + r[..., (1,)]
    )
    AB = (a004 * c2 + r[..., 2:]).clamp(min=0.0, max=1.0)
    return AB


def saturate_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True).clamp(min=1e-4, max=1.0)


def aces_film(rgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    EPS = 1e-6
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    rgb = (rgb * (a * rgb + b)) / (rgb * (c * rgb + d) + e)
    if isinstance(rgb, np.ndarray):
        return rgb.clip(min=0.0, max=1.0)
    elif isinstance(rgb, torch.Tensor):
        return rgb.clamp(min=0.0, max=1.0)
    