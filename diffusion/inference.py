import argparse
import os
import torch

from model import DiffusionModel
from unet import Unet
from torchvision.utils import save_image
from cleanfid import fid as cleanfid

@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    ##################################################################
    # TODO 3.3: Write a function that samples images from the
    # diffusion model given z
    # Note: The output must be in the range [0, 255]!
    ##################################################################
    def gen_fn(z):
        """
        Generator function that takes latent noise z and returns images in [0, 255] range.
        
        Args:
            z: Latent noise tensor of shape (batch_size, z_dimension)
        
        Returns:
            images: Generated images in range [0, 255] with shape (batch_size, 3, H, W)
        """
        # Reshape z to image dimensions (batch_size, channels, height, width)
        batch_size = z.shape[0]
        channels = 3  # CIFAR-10 has 3 color channels
        height = width = dataset_resolution  # 32x32 for CIFAR-10
        img_shape = (batch_size, channels, height, width)
        
        # Generate samples using the diffusion model with given z
        # The sample_given_z method generates images from provided noise
        generated_images = gen.sample_given_z(z, img_shape)
        
        # Convert from [0, 1] range to [0, 255] range
        # The diffusion model outputs images in [0, 1] after unnormalization
        images_255 = generated_images * 255.0
        
        # Clamp to ensure values are exactly in [0, 255]
        images_255 = torch.clamp(images_255, 0, 255)
        
        return images_255
    
    gen_fn = gen_fn
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    score = cleanfid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="train",
    )
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diffusion Model Inference')
    parser.add_argument('--ckpt', required=True, type=str, help="Pretrained checkpoint")
    parser.add_argument('--num-images', default=100, type=int, help="Number of images per iteration")
    parser.add_argument('--image-size', default=32, type=int, help="Image size to generate")
    parser.add_argument('--sampling-method', choices=['ddpm', 'ddim'])
    parser.add_argument('--ddim-timesteps', type=int, default=25, help="Number of timesteps to sample for DDIM")
    parser.add_argument('--ddim-eta', type=int, default=1, help="Eta for DDIM")
    parser.add_argument('--compute-fid', action="store_true")
    args = parser.parse_args()

    prefix = f"data_{args.sampling_method}/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    sampling_timesteps = args.ddim_timesteps if args.sampling_method == "ddim" else None

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).cuda()
    diffusion = DiffusionModel(
        model,
        timesteps=1000,   # number of timesteps
        sampling_timesteps=sampling_timesteps,
        ddim_sampling_eta=args.ddim_eta,
    ).cuda()

    img_shape = (args.num_images, diffusion.channels, args.image_size, args.image_size)

    # load pre-trained weight
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt["model_state_dict"])

    with torch.no_grad():
        # run inference
        model.eval()
        if args.sampling_method == "ddpm":
            generated_samples = diffusion.sample(img_shape)
        elif args.sampling_method == "ddim":
            generated_samples = diffusion.sample(img_shape)
        save_image(
            generated_samples.data.float(),
            prefix + f"samples_{args.sampling_method}.png",
            nrow=10,
        )
        if args.compute_fid:
            # NOTE: This will take a very long time to run even though we are only doing 10K samples.
            score = get_fid(diffusion, "cifar10", 32, 32*32*3, batch_size=256, num_gen=10_000)
            print("FID: ", score)