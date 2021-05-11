import torch
import torchshow as ts
import svox
import matplotlib.pyplot as plt
import kornia

import matplotlib.pyplot as plt

from nerf_pl.datasets.llff import *

from util import sample_pixels, sample_rays, sample_pixels_edge_bias
from render_utils import NeRF_Renderer, SVOX_Renderer
from transform_utils import twist_to_matrix

class iNeRF(torch.nn.Module):
    def __init__(self, pose_repr="twist", sampling="canny", device="cuda"):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.device = device
        self.image_res = (400, 400)
        self.NeRF_renderer = NeRF_Renderer(self.image_res)
        self.svox_renderer = SVOX_Renderer(self.image_res)
        self.pose_repr = pose_repr
        self.sampling = sampling

        self.init_c2w = torch.vstack((self.NeRF_renderer.dataset[9]["c2w"],
                                      torch.tensor([0., 0., 0., 1.]))).to(self.device)
        if pose_repr == "matrix":
            # Use euler angles, yaw and pitch as parameters of 3x3 rotation matrix
            # Initialize c2w as look-at matrix to object center from random position
            self.register_parameter(name='camera_rot', param=torch.nn.Parameter(self.init_c2w[:3, :3]))
            self.register_parameter(name='camera_pos', param=torch.nn.Parameter(self.init_c2w[:3, -1]))
        elif pose_repr == "twist":
            self.register_parameter(name='camera_twi', param=torch.nn.Parameter(torch.tensor([0., 0., 0., 0., 0., 1.])))
            self.register_parameter(name='camera_rot', param=torch.nn.Parameter(torch.zeros(1)))
        self.to(device)

        self.vis = [[], []]


    def forward(self, c2w=None, num_pixel_samples=100, target_image=None):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        c2w = self.c2w() if c2w is None else c2w.to(self.device)
        # print(c2w)
        if self.sampling == "random":
            pixel_idxs = sample_pixels(num_pixel_samples, self.image_res).to(self.device)
        elif self.sampling == "canny":
            pixel_idxs = sample_pixels_edge_bias(num_pixel_samples, target_image, self.image_res).to(self.device)
        else:
            raise NotImplementedError(f"Invalid sample {self.sampling}")
        # pixel_idxs = sample_pixels_edge_bias(num_pixel_samples, target_image, self.image_res).to(self.device)
        rays = sample_rays(pixel_idxs, c2w, self.NeRF_renderer.dataset)
        return self.NeRF_renderer.render_rays(rays)[0], pixel_idxs

    def render(self, theta=None, image_res=None):
        c2w = self.c2w() if theta is None else theta.to(self.device)
        image_res = self.image_res if image_res is None else image_res
        return self.svox_renderer.render_image(c2w, image_res)[0]

    def c2w(self, yaw=None, pitch=None):
        if self.pose_repr == "matrix":
            c2w = torch.zeros((4, 4)).to(self.device)
            c2w[:3, :3] = self.camera_rot
            c2w[:3, -1] = self.camera_pos
            c2w[-1, -1] = 1
        elif self.pose_repr == "twist":
            c2w = twist_to_matrix(self.camera_twi, self.camera_rot) @ self.init_c2w
        elif self.pose_repr == "euler":
            if yaw is None:
                yaw = self.camera_yaw
            if pitch is None:
                pitch = self.camera_pitch

            w2w_rot = torch.eye(4, device=device)
            w2w_rot[:3, :3] = yaw_pitch_to_rotation_matrix(yaw, pitch)[0]
            c2c_dist =torch.eye(4, device=device)
            c2c_dist[2, -1] = self.camera_dist
            c2w =  w2w_rot @ self.init_c2w_ @ c2c_dist
        else:
            raise NotImplementedError(f"Invalid pose_repr {self.pose_repr}")
        return c2w

    """
    TODO: Visualization - Want a set of plots and images that summarize a run's dynamics
      - Sphere with random sample pose points on it, can:
        - Visualize loss surface over all inward-facing directions
        - Visualize camera position trajectory
      - Grid of active candidate camera positions
      - Plots of: view-alignment loss, pose-estimation error
      - Show vector field of how each sample point "votes" or "pushes" guess pose towards target
    
    TODO: Optimization
      - Initialization: Use random sampling around sphere to find a good (set?) of initial poses
            - Is there a symmetry from
      - Bayesian optimization: https://distill.pub/2020/bayesian-optimization
      - Population / evolutionary optimization
      - Computing pixel sampling pdf using svox-rendered guess-view AND/or target view
    
    TODO Sampling
      - Actual PDF sampling
      - Pdf blur vs no blur
      - Hard negative mining -> seek pixels with differences (favor high magnitude pixels mismatches waay over many matches)
      - use importance sampling over target image pdf and guess image sample (integrate via averaging [union] or multiplication [intersection])
    
    TODO: Curiosities
      - Which degrees of freedom are hardest for the model to optimize over?
        - Look at different "slices" of optimization, where some params are fixed
          - Fix distance: produce loss volume over sphere or SO(3) 
          - Fix orientation (one free rotation dimension): produce loss volume enclosed by two circles with shared center
          - Try allowing roll variation
      - Could this be considered a pose refinement strategy? consider success rates for grid of (init distance X sample rate)
    """
    def fit(self, target_image):
        # Initialization Prologue
        if self.pose_repr == "euler":
            with torch.no_grad():
                n = 10
                best_loss = float('inf')
                best_yaw_pitch = None
                for pitch in torch.arange(0, np.pi / 3,
                                          1 / 3 * np.pi / 6):
                    for yaw in torch.arange(-np.pi, np.pi,
                                            2 * np.pi / 10):
                        c2w = self.c2w(yaw, pitch)

                        pred_pixels, pixel_idxs = self.forward(c2w, target_image=target_image.to(self.device))
                        target_pixels = target_image[pixel_idxs[:, 0], pixel_idxs[:, 1]]
                        loss = ((pred_pixels - target_pixels) ** 2).mean()

                        if loss < best_loss:
                            best_loss = loss
                            best_yaw_pitch = (yaw, pitch)
                print(best_yaw_pitch)
                self.register_parameter(name='camera_yaw', param=torch.nn.Parameter(torch.tensor([best_yaw_pitch[0] + 1e5])))
                self.register_parameter(name='camera_pitch', param=torch.nn.Parameter(torch.tensor([best_yaw_pitch[1] + 1e5])))
                # self.register_parameter(name='camera_dist', param=torch.nn.Parameter(torch.tensor([1e-5])))

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        for i in range(200):
            # Forward pass: Compute predicted y by passing x to the model
            pred_pixels, pixel_idxs = self.forward(target_image=target_image.to(self.device))
            target_pixels = target_image[pixel_idxs[:, 0], pixel_idxs[:, 1]]

            loss = ((pred_pixels - target_pixels)**2).mean()

            if i % 10 == 0:
                current_img = self.render().detach().cpu()
                self.vis[0].append((target_image.detach().cpu() + current_img * 2) / 3)
                self.vis[1].append(vis_pixies(pred_pixels, pixel_idxs, self.image_res))
            if i % 20 == 0:
                # if i == 0:
                #     vis_pixies(target_pixels, pixel_idxs, self.image_res)
                print(i, loss.item())

                # fig = plt.figure()
                # plt.imshow(current_img)
                # fig.suptitle(f"Pred image {i}")
                # plt.show()

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

        self.vis[0] = torch.stack(self.vis[0], dim=0).permute(0, 3, 1, 2)
        self.vis[1] = torch.stack(self.vis[1], dim=0).permute(0, 3, 1, 2)
        video = ts.show_video(self.vis, display=True)
        video.save("out.gif")

def vis_pixies(pred_pixels, pixel_idxs, wh=(800, 800), device='cuda'):
    img = torch.zeros((*wh, 3)).to(device)
    img[pixel_idxs[..., 0], pixel_idxs[..., 1]] = pred_pixels
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    img = kornia.box_blur(img * wh[0]//30 * wh[1]//30, (wh[0]//30, wh[1]//30), 'constant', normalized=False)
    # fig = plt.figure()
    # plt.imshow(img[0].permute(1, 2, 0).detach().cpu())
    # plt.savefig('canny_sampled.png')
    # fig.suptitle("Target image")
    # plt.show()
    return img[0].permute(1, 2, 0).detach().cpu()

if __name__ == "__main__":
    device = "cuda"
    with torch.no_grad():
        model = iNeRF(pose_repr="euler", sampling="canny")
        model.to(device)

        # Matrix copied from lego test set image 0
        c2w = torch.tensor([[-0.9999999403953552, 0.0, 0.0, 0.0],
                            [0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708],
                            [0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462],
                            [0.0, 0.0, 0.0, 1.0],
                            ], device=device)
        # c2w = torch.tensor([[-0.9999999403953552, 0.0, 0.0, 0.0],
        #                     [0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708],
        #                     [0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462],
        #                     [0.0, 0.0, 0.0, 1.0],
        #                     ], device=device)
        target_image = model.render(c2w)

    # fig = plt.figure()
    # plt.imshow(target_image.cpu())
    # fig.suptitle("Target image")
    # plt.savefig('out.png')
    # # plt.show()

    model.fit(target_image)
