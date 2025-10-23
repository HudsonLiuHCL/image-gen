import os
import torch
from utils import get_args
from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(discrim_real, discrim_fake, discrim_interp, interp, lamb):
    """
    WGAN-GP discriminator loss:
    loss = E[D(fake)] - E[D(real)] + λ * E[(||∇_x D(x̂)||₂ - 1)²]
    """
    # Part 1: Wasserstein loss
    loss_pt1 = torch.mean(discrim_fake) - torch.mean(discrim_real)

    # Part 2: Gradient penalty
    # Compute gradients of D(interpolated) w.r.t. the interpolated samples
    gradients = torch.autograd.grad(
        outputs=discrim_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(discrim_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Compute gradient norm
    grad_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)

    # Gradient penalty term
    grad_penalty = lamb * torch.mean((grad_norm - 1.0) ** 2)

    # Total loss
    loss = loss_pt1 + grad_penalty
    return loss


def compute_generator_loss(discrim_fake):
    """
    WGAN-GP generator loss:
    loss = -E[D(fake)]
    """
    loss = -torch.mean(discrim_fake)
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
