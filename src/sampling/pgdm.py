from typing import Tuple

import torch
from sampling.epsilon_net import ddim_step, EpsilonNetSVD


def pgdm_svd(
    initial_noise: torch.Tensor,
    inverse_problem: Tuple,
    epsilon_net: EpsilonNetSVD,
    eta: float = 1.0,
):
    """Use πGDM algorithm to solve an inverse problem.

    The algorithm operates on orthonormal basis defined by the SVD.
    Pass in the wrapper ``EpsilonNetSVD`` around ``EpsilonNet``.

    This is an implementation of [1].

    References
    ----------
    .. [1] Song, Jiaming, et al. "Pseudoinverse-guided diffusion models for inverse problems."
        International Conference on Learning Representations. 2023
    """
    obs, H_func, std_obs = inverse_problem

    # ensure obs has the right shape (1, -1)
    obs = obs.reshape(1, -1)
    Ut_y, diag = H_func.Ut(obs), H_func.singulars()

    alphas_cumprod = epsilon_net.alphas_cumprod
    timesteps = epsilon_net.timesteps

    def pot_fn(x, t):
        rsq_t = 1 - alphas_cumprod[t]
        diag_cov = diag**2 + (std_obs**2 / rsq_t)
        return (
            -0.5
            * torch.norm((Ut_y - diag * x[:, : diag.shape[0]]) / diag_cov.sqrt()) ** 2.0
        )

    sample = initial_noise.reshape(initial_noise.shape[0], -1)
    for i in range(len(timesteps) - 1, 1, -1):
        t, t_prev = timesteps[i], timesteps[i - 1]

        sample = sample.requires_grad_()
        xhat_0 = epsilon_net.predict_x0(sample, t)
        acp_t, acp_tprev = (
            torch.tensor([alphas_cumprod[t]]),
            torch.tensor([alphas_cumprod[t_prev]]),
        )

        grad_pot = pot_fn(xhat_0, t)
        grad_pot = torch.autograd.grad(grad_pot, sample)[0]
        sample = ddim_step(
            x=sample, epsilon_net=epsilon_net, t=t, t_prev=t_prev, eta=eta, e_t=xhat_0
        ).detach()
        sample += acp_tprev.sqrt() * acp_t.sqrt() * grad_pot

    # last diffusion step
    sample = epsilon_net.predict_x0(sample, timesteps[1])

    # map back to original pixel space
    sample = H_func.V(sample).reshape(initial_noise.shape)
    sample = sample.detach()

    return sample
