import time
from nerf_pl.models.rendering import *
from nerf_pl.models.nerf import *
import nerf_pl.metrics
from nerf_pl.datasets import dataset_dict
import torch
from nerf_pl.utils import *
from collections import defaultdict
import svox
from nerf_pl.datasets.ray_utils import get_ray_directions, get_rays
import matplotlib.pyplot as plt


class Renderer:
    def __init__(self, image_res):
        self.image_res = image_res
    def render_rays(self, **kwargs): raise NotImplementedError()
    def render_image(self, **kwargs): raise NotImplementedError()


class NeRF_Renderer(Renderer):
    """
    Wrapper for NeRF model(s)
    """
    def __init__(self, image_res, device="cuda"):
        super().__init__(image_res)
        self.device = device
        self.dataset = dataset_dict['blender']('data\\nerf_synthetic\\lego', 'test', img_wh=self.image_res)

        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)

        self.nerf_coarse = NeRF()
        self.nerf_fine = NeRF()

        ckpt_path = 'ckpts/lego.ckpt'

        load_ckpt(self.nerf_coarse, ckpt_path, model_name='nerf_coarse')
        load_ckpt(self.nerf_fine, ckpt_path, model_name='nerf_fine')

        self.nerf_coarse.to(device).eval()
        self.nerf_fine.to(device).eval()

        self.N_samples = 64
        self.N_importance = 64
        self.use_disp = False

    def render_rays(self, rays):
        results = render_rays([self.nerf_coarse, self.nerf_fine],
                              [self.embedding_xyz, self.embedding_dir],
                              rays.to(self.device), self.N_samples, self.use_disp,
                              perturb=0, noise_std=0,
                              N_importance=self.N_importance,
                              white_back=self.dataset.white_back,
                              test_time=False)
        return results['rgb_fine'], results

    def render_image(self, c2w):
        """Do batched inference on rays using chunk."""
        directions = self.dataset.directions.to(self.device)
        rays_o, rays_d = get_rays(directions, c2w.to(self.device))
        rays = torch.cat([
            rays_o,
            rays_d,
            self.dataset.near * torch.ones_like(rays_o[:, :1]),
            self.dataset.far * torch.ones_like(rays_o[:, :1])],
            1).to(self.device)  # (n_rays, 8)

        B = rays.shape[0]
        chunk = 1024*32
        results = defaultdict(list)
        for i in range(0, B, chunk):
            rendered_ray_chunks = \
                render_rays([self.nerf_coarse, self.nerf_fine],
                            [self.embedding_xyz, self.embedding_dir],
                            rays[i:i + chunk],
                            self.N_samples,
                            self.use_disp,
                            0,
                            0,
                            self.N_importance,
                            chunk,
                            self.dataset.white_back,
                            test_time=True)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results['rgb_fine'].reshape(*self.image_res, 3), results


class SVOX_Renderer(Renderer):
    """
    Wrapper for SVOX renderer
    """
    def __init__(self, image_res, device='cuda'):
        super().__init__(image_res)
        t = svox.N3Tree.load("ckpts/lego_sm.npz", map_location=device)
        self.renderer = svox.VolumeRenderer(t)

    def render_rays(self, rays):
        return self.renderer.forward(rays, cuda=True, fast=False), None

    def render_image(self, c2w, image_res=None):
        image_res = self.image_res if image_res is None else image_res
        focal_length = 1111.111 * image_res[0] / 800
        return self.renderer.render_persp(c2w,
                                          height=image_res[0],
                                          width=image_res[1],
                                          fx=focal_length, cuda=True, fast=True).clamp_(0.0, 1.0), None


if __name__ == "__main__":
    device = "cuda"

    image_res = (400, 400)
    render_NeRF = NeRF_Renderer(image_res)
    render_svox = SVOX_Renderer(image_res)

    with torch.no_grad():
        # Matrix copied from lego test set image 0
        c2w = torch.tensor([[-0.9999999403953552, 0.0, 0.0, 0.0],
                            [0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708],
                            [0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462],
                            [0.0, 0.0, 0.0, 1.0],
                            ], device=device)
        t = time.time()
        nerf_image, _ = render_NeRF.render_image(c2w)
        print("NeRF render speed", time.time() - t)
        t = time.time()
        svox_image, _ = render_svox.render_image(c2w)
        print("svox render speed", time.time() - t)
        target_image = torch.cat((nerf_image, svox_image), dim=1)

    fig = plt.figure()
    plt.imshow(target_image.cpu())
    fig.suptitle("Renderer image comparison")
    plt.show()
    print("Renderer difference", torch.sum(nerf_image - svox_image))
