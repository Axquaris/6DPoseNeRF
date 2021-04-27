import torch
import svox
import matplotlib.pyplot as plt
import kornia

import matplotlib.pyplot as plt

from nerf_pl.datasets.llff import *

from util import sample_pixels, sample_rays
from render_utils import NeRF_Renderer, SVOX_Renderer

# <(-.0)< <(0.-)> >(0.0)>

class iNeRF(torch.nn.Module):
    def __init__(self, device="cuda"):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.device = device
        self.image_res = (400, 400)
        self.NeRF_renderer = NeRF_Renderer(self.image_res)
        self.svox_renderer = SVOX_Renderer(self.image_res)

        init_c2w = self.NeRF_renderer.dataset[1]["c2w"]
        # Use euler angles, yaw and pitch as parameters of 3x3 rotation matrix
        # Initialize c2w as look-at matrix to object center from random position
        self.register_parameter(name='camera_rot', param=torch.nn.Parameter(init_c2w[:3, :3]))
        self.register_parameter(name='camera_pos', param=torch.nn.Parameter(init_c2w[:3, -1]))
        self.to(device)

    def forward(self, c2w=None, num_pixel_samples=800):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        c2w = self.c2w if c2w is None else c2w.to(self.device)
        pixel_idxs = sample_pixels(num_pixel_samples, self.image_res).to(self.device)
        rays = sample_rays(pixel_idxs, c2w, self.NeRF_renderer.dataset)
        return self.NeRF_renderer.render_rays(rays)[0], pixel_idxs

    def render(self, theta=None, image_res=None):
        c2w = self.c2w if theta is None else theta.to(self.device)
        image_res = self.image_res if image_res is None else image_res
        return self.svox_renderer.render_image(c2w, image_res)[0]

    @property
    def c2w(self):
        c2w = torch.zeros((4, 4)).to(self.device)
        c2w[:3, :3] = torch.eig(self.camera_rot, eigenvectors=True)[1]
        c2w[:3, -1] = self.camera_pos
        c2w[-1, -1] = 1
        return c2w

    def fit(self, target_image):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
        for i in range(2000):
            # Forward pass: Compute predicted y by passing x to the model
            pred_pixels, pixel_idxs = self.forward()
            target_pixels = target_image[pixel_idxs[:, 0], pixel_idxs[:, 1]]

            loss = ((pred_pixels - target_pixels)**2).sum()

            if i % 10 == 0:
                # if i ==0:
                #     vis_pixies(target_pixels, pixel_idxs, self.image_res)
                # vis_pixies(pred_pixels, pixel_idxs, self.image_res)

                print(i, loss.item())
                fig = plt.figure()
                plt.imshow(self.render().detach().cpu())
                fig.suptitle(f"Pred image {i}")
                plt.show()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Result: {model.string()}')

def vis_pixies(pred_pixels, pixel_idxs, wh=(800, 800), device='cuda'):
    img = torch.zeros((*wh, 3)).to(device)
    img[pixel_idxs[..., 0], pixel_idxs[..., 1]] = pred_pixels
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    img = kornia.box_blur(img * wh[0]//50 * wh[1]//50, (wh[0]//50, wh[1]//50), 'constant', normalized=False)
    fig = plt.figure()
    plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
    fig.suptitle("Target image")
    plt.show()

if __name__ == "__main__":
    device = "cuda"
    with torch.no_grad():
        model = iNeRF()
        model.to(device)

        # Matrix copied from lego test set image 0
        c2w = torch.tensor([[-0.9999999403953552, 0.0, 0.0, 0.0],
                            [0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708],
                            [0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462],
                            [0.0, 0.0, 0.0, 1.0],
                            ], device=device)
        target_image = model.render(c2w)

    fig = plt.figure()
    plt.imshow(target_image.cpu())
    fig.suptitle("Target image")
    plt.show()

    model.fit(target_image)
