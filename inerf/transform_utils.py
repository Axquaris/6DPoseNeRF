import kornia
import numpy as np
import torch
import cv2
from torch.nn import functional as F
from nerf_pl.datasets.ray_utils import get_ray_directions, get_rays
from functools import cache
from unittest import TestCase
import matplotlib.pyplot as plt
import time
from nerf_pl.models.rendering import *
from nerf_pl.models.nerf import *
import nerf_pl.metrics
from nerf_pl.datasets import dataset_dict
import torch
from render_utils import SVOX_Renderer
import torchshow as ts
from collections import defaultdict
import svox
from nerf_pl.datasets.ray_utils import get_ray_directions, get_rays
import matplotlib.pyplot as plt

device="cuda"

def twist_to_matrix(xi, theta, initial_c2w, device="cuda"):
    vee, omega = xi[:3], xi[3:]

    omega_hat = torch.zeros((3, 3)).to(device)
    omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0] = omega
    omega_hat[1, 2], omega_hat[2, 0], omega_hat[0, 1] = -omega
    omega_norm = torch.norm(omega)

    exp_omega = torch.eye(3).to(device) + omega_hat / omega_norm * torch.sin(omega_norm * theta) + \
        (omega_hat @ omega_hat) / (omega_norm ** 2) * (1 - torch.cos(omega_norm * theta))

    exp_xi = torch.zeros((4, 4)).to(device)
    exp_xi[:3, :3] = exp_omega
    exp_xi[:3, 3] = 1 / (omega_norm ** 2) * ((torch.eye(3).to(device) - exp_omega) @ (omega_hat @ vee) \
                                             + (torch.outer(omega, omega) @ vee) * theta)
    exp_xi[3, 3] = 1.0

    return exp_xi @ initial_c2w

# def pitch_yaw_to_rotation_matrix(pitch, yaw):
#     dir = torch.tensor([1, 0, 0]).to(device)
#     pitch = kornia.geometry.angle_axis_to_rotation_matrix(pitch.to(device) * dir.permute())
#     yaw = kornia.geometry.angle_axis_to_rotation_matrix(yaw.to(device) * dir)
#     return yaw @ pitch # apply yaw then pitch when going from camera->world->perturbed_world


# def matrix_to_pitch_yaw(pitch, yaw):
#     pass

# if __name__ == "__main__":
#     device = "cuda"
#
#     image_res = (400, 400)
#     render_svox = SVOX_Renderer(image_res)
#     image_list = []
#
#     with torch.no_grad():
#         # Matrix copied from lego test set image 0
#         c2w = torch.tensor([[-0.9999999403953552, 0.0, 0.0, 0.0],
#                             [0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708],
#                             [0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462],
#                             [0.0, 0.0, 0.0, 1.0],
#                             ], device=device)
#         rot = pitch_yaw_to_rotation_matrix(torch.Tensor([0]), torch.Tensor([0]))
#         svox_image, _ = render_svox.render_image(c2w @ rot)
#         image_list.append(svox_image)
#
#     image_list = torch.stack(image_list, dim=0).permute(0, 3, 1, 2)
#     video = ts.show_video(image_list, display=True)
#     video.save("360.gif")
