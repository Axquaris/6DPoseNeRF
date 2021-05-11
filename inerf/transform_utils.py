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

def twist_to_matrix(xi, theta, device="cuda"):
    vee, omega = xi[:3], xi[3:]

    omega_hat = torch.zeros((3, 3)).to(device)
    omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0] = omega
    omega_hat[1, 2], omega_hat[2, 0], omega_hat[0, 1] = -omega
    omega_norm = torch.norm(omega)

    exp_omega = torch.eye(3).to(device) + omega_hat / omega_norm * torch.sin(omega_norm * theta)
    exp_omega += (omega_hat @ omega_hat) / (omega_norm ** 2) * (1 - torch.cos(omega_norm * theta))

    exp_xi = torch.zeros((4, 4)).to(device)
    exp_xi[:3, :3] = exp_omega
    exp_xi[:3, 3] = 1 / (omega_norm ** 2) * ((torch.eye(3).to(device) - exp_omega) @ (omega_hat @ vee) \
                                             + (torch.outer(omega, omega) @ vee) * theta)
    exp_xi[3, 3] = 1.0

    return exp_xi

def yaw_pitch_to_rotation_matrix(yaw, pitch, device="cuda"):
    up = torch.tensor([[0, 0, 1]]).to(device)
    yaw_m = kornia.geometry.angle_axis_to_rotation_matrix(yaw.to(device) * up)

    right = torch.tensor([[-1, 0, 0]]).to(device)
    pitch_m = kornia.geometry.angle_axis_to_rotation_matrix(pitch.to(device) * right)
    return yaw_m @ pitch_m # apply yaw then pitch when going from camera->world->perturbed_world


# def matrix_to_pitch_yaw(pitch, yaw):
#     pass

if __name__ == "__main__":
    device = "cuda"

    image_res = (400, 400)
    render_svox = SVOX_Renderer(image_res)
    image_list = []

    with torch.no_grad():
        n = 10
        for pitch in torch.arange(0, np.pi / 3,
                                  1 / 3 * np.pi / 6):
            for yaw in torch.arange(-np.pi, np.pi,
                                    2 * np.pi / 10):
                # Matrix copied from lego test set image 0
                # c2w_init = torch.tensor([[-0.9999999403953552, 0.0, 0.0, 0.0],
                #                          [0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708],
                #                          [0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462],
                #                          [0.0, 0.0, 0.0, 1.0],
                #                          ], device=device)
                c2w_init = torch.tensor([[1., 0., 0., 0.],
                                         [0., 1., 0., .3],
                                         [0., 0., 1., 3.3 + 5],
                                         [0., 0., 0., 1.],
                                         ], device=device)
                init_correction = torch.eye(4, device=device)
                init_correction[:3, :3] = yaw_pitch_to_rotation_matrix(torch.Tensor([0]), torch.Tensor([-np.pi/2]))[0]
                w2w_rot = torch.eye(4, device=device)
                w2w_rot[:3, :3] = yaw_pitch_to_rotation_matrix(yaw, pitch)[0]

                svox_image, _ = render_svox.render_image(w2w_rot @ init_correction @ c2w_init)

                # plt.imshow(svox_image.cpu())
                # plt.show()
                # exit()
                image_list.append(svox_image)

    image_list = torch.stack(image_list, dim=0).permute(0, 3, 1, 2)
    video = ts.show_video(image_list, display=True)
    video.save("360.gif")
