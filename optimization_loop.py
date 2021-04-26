# -*- coding: utf-8 -*-
import torch
import math
import svox
from pytorch3d import transforms
import matplotlib as plt
import kornia

class ImageCoords(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(()))
        self.y = torch.nn.Parameter(torch.randn(()))
        self.z = torch.nn.Parameter(torch.randn(()))
        self.r = torch.nn.Parameter(torch.randn(()))
        self.p = torch.nn.Parameter(torch.randn(()))
        self.y = torch.nn.Parameter(torch.randn(()))

        t = svox.N3Tree(map_location="cpu")
        self.renderer = svox.VolumeRenderer(t)

    def forward(self, xi):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        c2w = transforms.rotation_6d_to_matrix(xi)
        return self.renderer.render_persp(c2w, height=800, width=800, fx=1111.111)



    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'x = {self.x}, y = {self.y}, z = {self.z}, r = {self.r}, p = {self.p}, y = {self.y}'

def loop():
    # Create Tensors to hold input and outputs.
    # x = torch.linspace(-math.pi, math.pi, 2000)
    # y = torch.sin(x)

    # does x hold 6d tensor, and y hold target image object?

    # Construct our model by instantiating the class defined above
    model = ImageCoords()

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters (defined
    # with torch.nn.Parameter) which are members of the model.
    # criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
    for t in range(2000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = kornia.psnr_loss(y_pred, y) # experiment with these
        if t % 100 == 99:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Result: {model.string()}')

if __name__ == "__main__":
    # loop()

    ## BELOW IS GREGORY'S TEST
    device = "cpu"

    # t = svox.N3Tree.load("lego_sm.npz", map_location=device)
    t = svox.N3Tree(map_location=device)
    r = svox.VolumeRenderer(t)

    # Matrix copied from lego test set image 0
    c2w = torch.tensor([[-0.9999999403953552, 0.0, 0.0, 0.0],
                        [0.0, -0.7341099977493286, 0.6790305972099304, 2.737260103225708],
                        [0.0, 0.6790306568145752, 0.7341098785400391, 2.959291696548462],
                        [0.0, 0.0, 0.0, 1.0],
                        ], device=device)

    with torch.no_grad():
        im = r.render_persp(c2w, height=800, width=800, fx=1111.111).clamp_(0.0, 1.0)
    plt.imshow(im.cpu())
    plt.show()