from typing import Tuple

import torch
from sampling.epsilon_net import ddim_step, EpsilonNetSVD


def repaint_svd(
    initial_noise: torch.Tensor,
    inverse_problem: Tuple,
    epsilon_net: EpsilonNetSVD,
    n_reps: int = 2,
    eta: float = 1.0,
):
    """Use RePaint algorithm to solve an inverse problem.

    This is a modified version of the original algorithm [1]
    to solve general Linear inverse problems.

    The algorithm operates on the orthonormal basis defined by the SVD.
    Pass in the wrapper ``EpsilonNetSVD`` around ``EpsilonNet``.

    References
    ----------
    .. [1] Lugmayr, Andreas, et al. "Repaint: Inpainting using denoising diffusion probabilistic models."
        Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
    """
    obs, H_func, std_obs = inverse_problem

    # ensure obs has the right shape (1, -1)
    obs = obs.reshape(1, -1)
    Ut_y, diag = H_func.Ut(obs), H_func.singulars()

    alphas_cumprod = epsilon_net.alphas_cumprod
    timesteps = epsilon_net.timesteps

    sample = initial_noise.reshape(initial_noise.shape[0], -1)
    for i in range(len(timesteps) - 1, 1, -1):
        t, t_prev = timesteps[i], timesteps[i - 1]

        for r in range(n_reps):

            acp_t, acp_tprev = (
                torch.tensor([alphas_cumprod[t]]),
                torch.tensor([alphas_cumprod[t_prev]]),
            )

            sample = ddim_step(
                x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta
            )

            # replace with know value
            # don't forget rescale by the diag
            noise = torch.randn_like(Ut_y)
            sample[:, : diag.shape[0]] = (
                acp_tprev.sqrt() * Ut_y / diag + (1 - acp_tprev).sqrt() * noise
            )

            # Don't get back to x_t in the last rep
            if r != n_reps - 1:
                a_t = acp_t / acp_tprev
                noise = torch.randn_like(sample)
                sample = a_t.sqrt() * sample + (1 - a_t).sqrt() * noise

    # last diffusion step
    sample = epsilon_net.predict_x0(sample, timesteps[1])

    # map back to original pixel space
    sample = H_func.V(sample).reshape(initial_noise.shape)
    sample = sample.detach()

    return sample
