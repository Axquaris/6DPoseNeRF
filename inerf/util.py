import kornia
import numpy as np
import torch
import cv2
from torch.nn import functional as F
from nerf_pl.datasets.ray_utils import get_ray_directions, get_rays
from functools import cache
from unittest import TestCase
import matplotlib.pyplot as plt

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def rotation_9d_to_matrix(d9: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    rot_mat = rotation_6d_to_matrix(d9[..., 3:])
    b = torch.reshape(d9[...,:3], (3, 1))
    new_mat = torch.cat((rot_mat, b), dim=1)
    new_mat = torch.cat((new_mat, torch.tensor([[0, 0, 0, 1]], dtype=float)), dim=0)
    return new_mat


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


@cache
def sampling_dist(target_image):
    np_img = np.array(target_image * 255, dtype=np.uint8)
    edge_img = cv2.Canny(np_img, 60, 120)
    non_black = np.nonzero(edge_img)
    edge_pixels = np.array((non_black[0], non_black[1])).T
    return torch.as_tensor(edge_pixels, dtype=torch.long)

def sample_pixels_edge_bias(n_pixels, target_image, image_res=(800, 800)):
    pixel_options = sampling_dist(target_image)
    rand_idxs = torch.randperm(pixel_options.shape[0])[:n_pixels]
    return pixel_options[rand_idxs]

def sample_pixels(n_pixels, image_res=(800, 800)):
    """
    Sample a set of pixels
    :param image_res: (2) image dimensions
    :return: (n_rays, 2) tensor containing pixels to be sampled
    """
    all_pixels = kornia.create_meshgrid(*image_res, normalized_coordinates=False)[0].type(torch.long).reshape(-1, 2)
    rand_idxs = torch.randperm(all_pixels.shape[0])[:n_pixels]
    return all_pixels[rand_idxs]

def sample_rays(pixels, c2w, dataset, device='cuda'):
    """
    Given a batch of pixels, calculate rays to be sampled for each of them
    focal length is assumed to be 1, sampling every possible ray atm, should optimize
    :param pixels: (n_rays, 2) tensor containing pixels to be sampled
    :param image_res: (2) image dimensions
    :return: (n_rays, 8) tensor containing rays through each pixel, [origin, direction, t_min, t_max]
    """

    # hFov, vFov to translate to camera coord system
    # c2w matrix goes from camera to world coord system
    # camera plane is at z = 1, with size of plane = image_res
    sampled_dirs = dataset.directions[pixels[...,0], pixels[...,1]].to(device)
    rays_o, rays_d = get_rays(sampled_dirs, c2w)
    return_mat = torch.cat([
        rays_o,
        rays_d,
        dataset.near*torch.ones_like(rays_o[:, :1]),
        dataset.far*torch.ones_like(rays_o[:, :1])],
        1).to(device) # (n_rays, 8)
    return return_mat


class Tests(TestCase):
    @staticmethod
    def test_sample_pixels_edge_bias():
        from nerf_pl.datasets import dataset_dict
        dataset = dataset_dict['blender']('data\\nerf_synthetic\\lego', 'test', img_wh=(800, 800))
        sample = dataset[0]
        # print(list(sample.keys()))
        img = sample['rgbs'].reshape(800, 800, 3)
        samples = sample_pixels_edge_bias(800*800//10, img)
        img = img * .4
        for pixl in samples:
            img[pixl[0], pixl[1]] = torch.Tensor([1,0,0])
        plt.imshow(img.detach().cpu())
        plt.show()

    @staticmethod
    def test_sample_rays():
        from nerf_pl.datasets import dataset_dict
        dataset = dataset_dict['blender']('data\\nerf_synthetic\\lego', 'test', img_wh=(800, 800))
        sample = dataset[0]

        pixels = sample_pixels(1000)

        guess_rays = sample_rays(pixels, sample["c2w"], dataset)
        target_rays = sample['rays'].cuda()
        target_rays = target_rays[pixels[..., 0] * 800 + pixels[..., 1]]

        print(torch.sum(guess_rays - target_rays))


if __name__ == "__main__":
    Tests.test_sample_pixels_edge_bias()
