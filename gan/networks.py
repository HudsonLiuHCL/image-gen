import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.upscale_factor = upscale_factor

    @torch.jit.script_method
    def forward(self, x):

        r = self.upscale_factor                     
        x = x.repeat(1, r * r, 1, 1)                
        x = F.pixel_shuffle(x, r)                   
        out = self.conv(x)
        return out


class DownSampleConv2D(torch.jit.ScriptModule):
    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        self.conv = nn.Conv2d(
            input_channels, n_filters, kernel_size=kernel_size, padding=padding
        )
        self.downscale_ratio = downscale_ratio

    @torch.jit.script_method
    def forward(self, x):

        r = self.downscale_ratio                     
        x = F.pixel_unshuffle(x, r)                  
        B, Cr2, H, W = x.shape
        C = Cr2 // (r * r)
        x = x.view(B, r * r, C, H, W)                 
        x = x.mean(dim=1)                             
        out = self.conv(x)
        return out




class ResBlockUp(torch.jit.ScriptModule):
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            UpSampleConv2D(
                n_filters, kernel_size=kernel_size, n_filters=n_filters, upscale_factor=2, padding=1
            ),
        )

        self.upsample_residual = UpSampleConv2D(
            input_channels, kernel_size=1, n_filters=n_filters, upscale_factor=2, padding=0
        )

    @torch.jit.script_method
    def forward(self, x):
        out = self.layers(x)                 # main path
        residual = self.upsample_residual(x) # shortcut path
        return out + residual                # combine



class ResBlockDown(torch.jit.ScriptModule):
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            DownSampleConv2D(
                n_filters, kernel_size=kernel_size, n_filters=n_filters, downscale_ratio=2, padding=1
            ),
        )

        self.downsample_residual = DownSampleConv2D(
            input_channels, kernel_size=1, n_filters=n_filters, downscale_ratio=2, padding=0
        )

    @torch.jit.script_method
    def forward(self, x):
        out = self.layers(x)                
        residual = self.downsample_residual(x)  
        return out + residual                



class ResBlock(torch.jit.ScriptModule):
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, n_filters, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, padding=1),
        )

    @torch.jit.script_method
    def forward(self, x):
        out = self.layers(x)  
        return out + x        


import torch
import torch.nn as nn

class Generator(torch.jit.ScriptModule):
    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        ##################################################################
        # 1️⃣ Dense layer: map noise vector z (128D) → 4×4×128 feature map
        ##################################################################
        self.latent_dim = 128
        self.starting_image_size = starting_image_size
        self.feature_channels = 128

        self.dense = nn.Linear(
            self.latent_dim,
            self.feature_channels * starting_image_size * starting_image_size
        )

        ##################################################################
        # 2️⃣ Build upsampling layers using ResBlockUps
        # Each ResBlockUp doubles spatial resolution
        ##################################################################
        self.layers = nn.Sequential(
            ResBlockUp(self.feature_channels, n_filters=self.feature_channels),  # 4×4 → 8×8
            ResBlockUp(self.feature_channels, n_filters=self.feature_channels),  # 8×8 → 16×16
            ResBlockUp(self.feature_channels, n_filters=self.feature_channels),  # 16×16 → 32×32
            nn.BatchNorm2d(self.feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feature_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # output in [-1, 1] range
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward_given_samples(self, z):
        ##################################################################
        # 3️⃣ Given a latent vector z → generate an image
        ##################################################################
        # Flatten → Dense → Reshape → Upsample through layers
        out = self.dense(z)
        out = out.view(-1, self.feature_channels, self.starting_image_size, self.starting_image_size)
        out = self.layers(out)
        return out
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        ##################################################################
        # 4️⃣ Auto-generate n_samples of random z and forward through generator
        ##################################################################
        device = list(self.parameters())[0].device
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.forward_given_samples(z)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################



class Discriminator(torch.jit.ScriptModule):
    def __init__(self):
        super(Discriminator, self).__init__()
        ##################################################################
        # 1️⃣ Define network layers
        ##################################################################
        self.layers = nn.Sequential(
            # 32×32×3 → 16×16×128
            ResBlockDown(3, n_filters=128),
            # 16×16×128 → 8×8×128
            ResBlockDown(128, n_filters=128),
            # Two same-size residual blocks
            ResBlock(128, n_filters=128),
            ResBlock(128, n_filters=128),
            nn.ReLU(inplace=True),
        )

        # Final dense layer: compress 128 channels → 1 scalar output
        self.dense = nn.Linear(128, 1)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    @torch.jit.script_method
    def forward(self, x):
        ##################################################################
        # 2️⃣ Forward pass
        # Input: x = (batch, 3, 32, 32)
        ##################################################################
        out = self.layers(x)  # shape → (batch, 128, 8, 8)
        out = out.sum(dim=[2, 3])  # spatial sum → (batch, 128)
        out = self.dense(out)      # → (batch, 1)
        return out
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
