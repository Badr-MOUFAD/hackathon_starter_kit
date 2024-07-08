import torch

from utils import display_image
from sampling.epsilon_net import EpsilonNet, bridge_kernel_statistics

import matplotlib.pyplot as plt


def unconditional_sampling(
    epsilon_net: EpsilonNet,
    initial_noise: torch.Tensor,
    eta: float = 1.0,
    display_im: bool = False,
    display_freq: int = 30,
):
    """Perform unconditional sampling with a diffusion model.

    Runs the algorithm in DDIM [1].
    ``eta`` controls the variance of the backward kernel.

    Note
    ----
    - Use ``initial_noise`` to set the number of samples ``(n_samples, *shape_of_data)``.
    - Use ``display_im`` and ``display_freq`` to plot the the evolution of samples throughout
    the sampling

    References
    ----------
    .. [1] Song, Jiaming, Chenlin Meng, and Stefano Ermon.
    "Denoising diffusion implicit models." arXiv preprint arXiv:2010.02502 (2020).
    """
    # init
    epsilon_net.requires_grad_(False)
    bridge_func = _build_bridge(epsilon_net, eta=eta)
    device = initial_noise.device

    x_t = torch.randn_like(initial_noise)
    timesteps = epsilon_net.timesteps

    range_sampling = range(len(timesteps) - 1, 1, -1)

    for i in range_sampling:
        t, t_prev = timesteps[i], timesteps[i - 1]
        x_t_prev, std_t = bridge_func(x_t, t, t_prev)

        x_t = x_t_prev + std_t * torch.randn_like(initial_noise, device=device)

        # plot evolution of sampling
        if display_im and i % display_freq == 0:
            _, axes = plt.subplots(1, 2)
            x_hat_0 = epsilon_net.predict_x0(x_t[[0]], t_prev)

            for ax, img, title in zip(axes, (x_hat_0, x_t[0]), ("x_0 | x_t", "x_t")):
                display_image(img, ax)
                ax.set_title(title)

    samples = epsilon_net.predict_x0(x_t, timesteps[1])

    return samples


def _build_bridge(epsilon_net: EpsilonNet, eta: float = 1.0):
    # wrapper around ``bridge_kernel_statistics``

    def bridge_func(x_t: torch.Tensor, t: int, t_prev: int, x_0t: torch.Tensor = None):
        t_0 = epsilon_net.timesteps[0]
        x_0t = epsilon_net.predict_x0(x_t, t) if x_0t is None else x_0t

        mean, std = bridge_kernel_statistics(
            x_ell=x_t,
            x_s=x_0t,
            epsilon_net=epsilon_net,
            ell=t,
            t=t_prev,
            s=t_0,
            eta=eta,
        )
        return mean, std

    return bridge_func
