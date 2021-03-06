import kornia
import numpy as np
import torch
import cv2
from torch.nn import functional as F
from nerf_pl.datasets.ray_utils import get_ray_directions, get_rays
from functools import cache
from unittest import TestCase
import matplotlib.pyplot as plt


@cache
def sampling_dist_new(target_image):
    np_img = np.array(target_image * 255, dtype=np.uint8)
    edge_img = cv2.Canny(np_img, 60, 120)
    blurred_img = cv2.GaussianBlur(edge_img,(15,15),0)
    norm_image = cv2.normalize(blurred_img, None, norm_type=cv2.NORM_MINMAX)
    weights = []
    for u in range(blurred_img.shape[0]):
        for v in range((blurred_img.shape[1])):
            weights.append((u, v, norm_image[u, v, 0]))
    weights = np.array(weights)
    weights[:, 2] = np.cumsum(weights[:, 2]/np.linalg.norm(weights[:, 2]))
    return torch.as_tensor(weights, dtype=torch.float)

@cache
def sampling_dist(target_image):
    np_img = np.array(target_image.to('cpu') * 255, dtype=np.uint8)
    edge_img = cv2.Canny(np_img, 60, 120)
    non_black = np.nonzero(edge_img)
    edge_pixels = np.array((non_black[0], non_black[1])).T
    return torch.as_tensor(edge_pixels, dtype=torch.long).to('cuda')

def sample_pixels_edge_bias(n_pixels, target_image, image_res=(800, 800)):
    pixel_options = sampling_dist(target_image)
    rand_idxs = torch.randperm(pixel_options.shape[0])[:n_pixels]
    return pixel_options[rand_idxs]

    # pixel_options = sampling_dist(target_image.cpu())
    # rand_idxs = np.random.choice(len(pixel_options), n_pixels, replace=False, p=pixel_options[:,2]/np.linalg.norm(pixel_options[:,2]))
    # return pixel_options[:,0:1][rand_idxs]

    # pixel_options = sampling_dist_new(target_image.cpu())
    # rand_vals = torch.rand(n_pixels)
    # rand_idx = (pixel_options[:, 2] < rand_vals).nonzero(as_tuple=True)[-1]
    # return pixel_options[:, 0:1][rand_idxs]

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
