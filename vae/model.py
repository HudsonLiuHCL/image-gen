import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim


class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.convs(dummy)
            flattened_dim = conv_out.numel()

        self.fc = nn.Linear(flattened_dim, latent_dim)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        z = self.fc(x)
        return z


class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        ##################################################################
        # Output should be twice the latent dimension: μ and logσ
        ##################################################################
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.convs(dummy)
            flattened_dim = conv_out.numel()

        self.fc = nn.Linear(flattened_dim, 2 * latent_dim)

    def forward(self, x):
        ##################################################################
        # TODO 2.1: Forward pass through the network, should return a
        # tuple of 2 tensors, mu and log_std
        ##################################################################
        x = self.convs(x)                   # pass input through conv layers
        x = x.view(x.size(0), -1)           # flatten
        out = self.fc(x)                    # linear projection (2 * latent_dim outputs)
        mu, log_std = torch.chunk(out, 2, dim=1)  # split into two halves
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return mu, log_std


class Decoder(nn.Module):
    """
    Sequential(
        (0): ReLU()
        (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (2): ReLU()
        (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (4): ReLU()
        (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (6): ReLU()
        (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    """
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        ##################################################################
        # TODO 2.1: Set up the network layers. First, compute
        # self.base_size, then create the self.fc and self.deconvs.
        ##################################################################
        # 32×32 input shrinks by stride-2 three times → 4×4 feature map
        self.base_size = 4
        self.fc = nn.Linear(latent_dim, 256 * self.base_size * self.base_size)

        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, stride=1, padding=1)
        )
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, z):
        ##################################################################
        # TODO 2.1: Forward pass through the network, first through
        # self.fc, then self.deconvs.
        ##################################################################
        x = self.fc(z)
        x = x.view(z.size(0), 256, self.base_size, self.base_size)
        x = self.deconvs(x)
        return x
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape=(3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    # NOTE: You don't need to implement a forward function for AEModel.
    # For implementing the loss functions in train.py, call model.encoder
    # and model.decoder directly.
