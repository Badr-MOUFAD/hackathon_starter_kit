import torch
import numpy as np
from torch import vmap
from torch.func import jacrev


class EpsilonNet(torch.nn.Module):
    def __init__(self, net, alphas_cumprod, timesteps):
        super().__init__()
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.timesteps = timesteps

    def forward(self, x, t):
        return self.net(x, torch.tensor(t))

    def predict_x0(self, x, t):
        alpha_cum_t = (
            self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0].int()]
        )
        return (x - (1 - alpha_cum_t) ** 0.5 * self.forward(x, t)) / (alpha_cum_t**0.5)

    def score(self, x, t):
        alpha_cum_t = self.alphas_cumprod[t] / self.alphas_cumprod[self.timesteps[0]]
        return -self.forward(x, t) / (1 - alpha_cum_t) ** 0.5

    def value_and_grad_predx0(self, x, t):
        x = x.requires_grad_()
        pred_x0 = self.predict_x0(x, t)
        grad_pred_x0 = torch.autograd.grad(pred_x0.sum(), x)[0]
        return pred_x0, grad_pred_x0

    def value_and_jac_predx0(self, x, t):
        def pred(x):
            return self.predict_x0(x, t)

        pred_x0 = self.predict_x0(x, t)
        return pred_x0, vmap(jacrev(pred))(x)


class EpsilonNetSVD(EpsilonNet):
    def __init__(self, net, alphas_cumprod, timesteps, H_func, device="cuda"):
        super().__init__(net, alphas_cumprod, timesteps)
        self.net = net
        self.alphas_cumprod = alphas_cumprod
        self.H_func = H_func
        self.timesteps = timesteps
        self.device = device

    def forward(self, x, t):
        shape = (x.shape[0], 3, int(np.sqrt((x.shape[-1] // 3))), -1)
        x = self.H_func.V(x.to(self.device)).reshape(shape)
        return self.H_func.Vt(self.net(x, t))


# -----------
# ----------- diffusion kernels
# -----------
def bridge_kernel_statistics(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNet,
    ell: int,
    t: int,
    s: int,
    eta: float,
):
    """s < t < ell"""
    alpha_cum_s_to_t = epsilon_net.alphas_cumprod[t] / epsilon_net.alphas_cumprod[s]
    alpha_cum_t_to_ell = epsilon_net.alphas_cumprod[ell] / epsilon_net.alphas_cumprod[t]
    alpha_cum_s_to_ell = epsilon_net.alphas_cumprod[ell] / epsilon_net.alphas_cumprod[s]
    std = (
        eta
        * ((1 - alpha_cum_t_to_ell) * (1 - alpha_cum_s_to_t) / (1 - alpha_cum_s_to_ell))
        ** 0.5
    )
    coeff_xell = ((1 - alpha_cum_s_to_t - std**2) / (1 - alpha_cum_s_to_ell)) ** 0.5
    coeff_xs = (alpha_cum_s_to_t**0.5) - coeff_xell * (alpha_cum_s_to_ell**0.5)
    return coeff_xell * x_ell + coeff_xs * x_s, std


def sample_bridge_kernel(
    x_ell: torch.Tensor,
    x_s: torch.Tensor,
    epsilon_net: EpsilonNet,
    ell: int,
    t: int,
    s: int,
    eta: float,
):
    mean, std = bridge_kernel_statistics(x_ell, x_s, epsilon_net, ell, t, s, eta)
    return mean + std * torch.randn_like(mean)


def ddim_statistics(
    x: torch.Tensor,
    epsilon_net: EpsilonNet,
    t: float,
    t_prev: float,
    eta: float,
    e_t: torch.Tensor = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t)
    return bridge_kernel_statistics(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def ddim_step(
    x: torch.Tensor,
    epsilon_net: EpsilonNet,
    t: float,
    t_prev: float,
    eta: float,
    e_t: torch.Tensor = None,
):
    t_0 = epsilon_net.timesteps[0]
    if e_t is None:
        e_t = epsilon_net.predict_x0(x, t)
    return sample_bridge_kernel(
        x_ell=x, x_s=e_t, epsilon_net=epsilon_net, ell=t, t=t_prev, s=t_0, eta=eta
    )


def ddim(
    initial_noise_sample: torch.Tensor, epsilon_net: EpsilonNet, eta: float = 1.0
) -> torch.Tensor:
    """
    This function implements the (subsampled) generation from https://arxiv.org/pdf/2010.02502.pdf (eqs 9,10, 12)

    Parameters
    ----------
    initial_noise_sample :
        Initial "noise"
    timesteps :
        List containing the timesteps. Should start by 999 and end by 0

    score_model :
        The score model

    eta :
        the parameter eta from https://arxiv.org/pdf/2010.02502.pdf (eq 16)
    """
    sample = initial_noise_sample
    for i in range(len(epsilon_net.timesteps) - 1, 1, -1):
        sample = ddim_step(
            x=sample,
            epsilon_net=epsilon_net,
            t=epsilon_net.timesteps[i],
            t_prev=epsilon_net.timesteps[i - 1],
            eta=eta,
        )
    return epsilon_net.predict_x0(sample, epsilon_net.timesteps[1])
