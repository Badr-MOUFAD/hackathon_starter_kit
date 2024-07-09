import numpy as np
from PIL import Image
import torch, PIL, yaml
from torch.distributions import Distribution

from diffusers import DDPMPipeline

from evaluation.gmm import EpsilonNetGM
from sampling.epsilon_net import EpsilonNet
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from local_paths import ABS_PROJECT_PATH

import matplotlib.pyplot as plt


def load_image(img_path, device="cpu") -> torch.Tensor:
    """Load and image as Tensor.

    img_path : str or Path
        the path of the image.
    """
    image = Image.open(img_path)

    im = torch.tensor(np.array(image)).type(torch.FloatTensor).to(device)
    x_origin = ((im - 127.5) / 127.5).squeeze(0)
    x_origin = x_origin.permute(2, 0, 1)

    return x_origin


def display_image(x: torch.Tensor, ax=None):
    """Display an image."""
    sample = x.squeeze(0).detach().cpu().permute(1, 2, 0)

    sample = sample.clamp(-1, 1)
    sample = (sample + 1.0) * 127.5
    sample = sample.numpy().astype(np.uint8)
    img_pil = PIL.Image.fromarray(sample)

    if ax is None:
        plt.imshow(img_pil)
    else:
        ax.imshow(img_pil)


def display_grayscale_image(x: torch.Tensor, ax=None):
    """Display a grayscale image."""
    sample = x.squeeze(0).detach().cpu()

    sample = sample.clamp(-1, 1)
    sample = (sample + 1.0) * 127.5
    sample = sample.numpy().astype(np.uint8)
    img_pil = PIL.Image.fromarray(sample)

    if ax is None:
        plt.imshow(img_pil, cmap="gray")
    else:
        ax.imshow(img_pil, cmap="gray")


def load_epsilon_net(model_id: str, n_steps: int, device: str) -> EpsilonNet:
    """Load a diffusion model with a given diffusion steps.

    model_id : str
        The model name and should be either "celebahq", "ffhq", "bedroom"

    n_steps : int
        The number of diffusion steps.

    device : str
        The device where to put the model.
    """


    HF_MODELS = {
        "bedroom": "google/ddpm-bedroom-256",
        "celebahq": "google/ddpm-celebahq-256"
    }

    if model_id in HF_MODELS:
        hf_id = HF_MODELS[model_id]
        pipeline = DDPMPipeline.from_pretrained(hf_id).to(device)
        model = pipeline.unet

        # by default set model to eval mode and disable grad on model parameters
        model = model.requires_grad_(False)
        model = model.eval()

        timesteps = torch.linspace(0, 999, n_steps, device=device).long()
        alphas_cumprod = pipeline.scheduler.alphas_cumprod.clip(1e-6, 1).to(device)
        alphas_cumprod = torch.concatenate(
            [torch.tensor([1.0], device=device), alphas_cumprod]
        )

        class UNet(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet

            def forward(self, x, t):
                if torch.tensor(t).dim() == 0:
                    t = torch.tensor([t])
                return self.unet(x, t).sample

        return EpsilonNet(
            net=UNet(model), alphas_cumprod=alphas_cumprod, timesteps=timesteps
        )

    elif model_id == "ffhq":
        # NOTE code verified at https://github.com/openai/guided-diffusion
        # and adapted from https://github.com/DPS2022/diffusion-posterior-sampling

        model_config = ABS_PROJECT_PATH / "/py_source/configs/ffhq_model.yaml"
        diffusion_config = ABS_PROJECT_PATH / "/py_source/configs/diffusion_config.yaml"

        # load configs
        with open(model_config) as f:
            model_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(diffusion_config) as f:
            diffusion_config = yaml.load(f, Loader=yaml.FullLoader)

        sampler = create_sampler(**diffusion_config)
        model = create_model(**model_config)

        # by default set model to eval mode and disable grad on model parameters
        model = model.eval()
        model.requires_grad_(False)

        timesteps = torch.linspace(0, 999, n_steps).long()
        alphas_cumprod = torch.tensor(sampler.alphas_cumprod).float().clip(1e-6, 1)
        alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

        class UNet(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet

            def forward(self, x, t):
                if torch.tensor(t).dim() == 0:
                    t = torch.tensor([t])
                return self.unet(x, t)[:, :3]

        return EpsilonNet(
            net=UNet(model),
            alphas_cumprod=alphas_cumprod,
            timesteps=timesteps,
        )
    else:
        raise ValueError(
            "Unknown model.\n" "`model_id` must be either 'celebahq' or 'ffhq' "
        )


def load_gmm_epsilon_net(prior: Distribution, dim: int, n_steps: int):
    timesteps = torch.linspace(0, 999, n_steps).long()
    alphas_cumprod = torch.linspace(0.9999, 0.98, 1000)
    alphas_cumprod = torch.cumprod(alphas_cumprod, 0).clip(1e-10, 1)
    alphas_cumprod = torch.concatenate([torch.tensor([1.0]), alphas_cumprod])

    means, covs, weights = (
        prior.component_distribution.mean,
        prior.component_distribution.covariance_matrix,
        prior.mixture_distribution.probs,
    )

    epsilon_net = EpsilonNet(
        net=EpsilonNetGM(means, weights, alphas_cumprod, covs),
        alphas_cumprod=alphas_cumprod,
        timesteps=timesteps,
    )

    return epsilon_net
