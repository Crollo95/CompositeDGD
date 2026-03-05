import torch
import torch.nn as nn
import torch.nn.functional as F

from .output_modules import (
    GaussianOutput,
    PoissonOutput,
    BernoulliOutput,
    BetaOutput,
    MultinomialOutput,
    NegativeBinomialOutput,
    SurvivalCPHOutput,
    CompositeOutput,
)

OUTPUT_MODULES = {
    "gaussian": GaussianOutput,
    "poisson": PoissonOutput,
    "bernoulli": BernoulliOutput,
    "beta": BetaOutput,
    "multinomial": MultinomialOutput,
    "nb": NegativeBinomialOutput,
    "survival_cph": SurvivalCPHOutput,
    "composite": CompositeOutput,
}


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims,
        out_dim: int,
        output_module: str = "gaussian",
        output_module_options: dict | None = None,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        layers = []
        in_dim = latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            in_dim = h
        self.net = nn.Sequential(*layers) if layers else nn.Identity()

        if output_module_options is None:
            output_module_options = {}

        if output_module not in OUTPUT_MODULES:
            raise ValueError(f"Unknown output_module '{output_module}'")

        OutputCls = OUTPUT_MODULES[output_module]

        # Some modules accept 'options', CompositeOutput needs it
        try:
            self.out = OutputCls(in_dim, out_dim, options=output_module_options)
        except TypeError:
            self.out = OutputCls(in_dim, out_dim)

    def forward(self, z: torch.Tensor):
        h = self.net(z)
        return self.out(h)

    def negative_log_likelihood(self, y: torch.Tensor, *out_params, scaling_factor=None):
        """
        out_params is whatever forward() returned:
          - GaussianOutput: (mean, log_var)
          - PoissonOutput: rate
          - CompositeOutput: head_out0, head_out1, ...
        """
        # Only NB needs scaling_factor; others ignore it.
        try:
            return self.out.loss(y, *out_params, scaling_factor=scaling_factor)
        except TypeError:
            return self.out.loss(y, *out_params)




import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderCNN(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        img_shape=(1, 28, 28),
        base_channels: int = 64,
        likelihood: str = "bernoulli",
        dropout_p: float = 0.0,
        film: bool = False,
        n_w: int | None = None,
    ):
        super().__init__()
        C, H, W = img_shape
        assert (H, W) == (28, 28)
        self.C, self.H, self.W = C, H, W

        self.likelihood = likelihood.lower()
        assert self.likelihood in {"bernoulli", "gaussian"}

        ch0 = base_channels * 2  # 128 if base_channels=64

        # --- FC block (with dropout ONLY here) ---
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, ch0 * 7 * 7)
        self.drop_fc = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # --- 7 -> 14 ---
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False) #mode="nearest")
        self.conv1a = nn.Conv2d(ch0, base_channels, 3, padding=1)
        self.conv1b = nn.Conv2d(base_channels, base_channels, 3, padding=1)  # extra conv

        # --- 14 -> 28 ---
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False ) #mode="nearest")
        self.conv2a = nn.Conv2d(base_channels, base_channels // 2, 3, padding=1)
        self.conv2b = nn.Conv2d(base_channels // 2, base_channels // 2, 3, padding=1)  # extra conv

        self.conv_out = nn.Conv2d(base_channels // 2, C, 3, padding=1)

        with torch.no_grad():
            # typical mean pixel for FashionMNIST is ~0.28-0.35; pick 0.3 as a default
            p0 = torch.tensor(0.30)
            b0 = float(torch.log(p0 / (1.0 - p0)))  # logit(p0)
            nn.init.constant_(self.conv_out.bias, b0)


        # --- FiLM conditioning (optional) ---
        self.film = film
        if film:
            assert n_w is not None, "n_w required when film=True"
            self.n_w = n_w
            self.film_7  = nn.Linear(n_w, ch0 * 2)
            self.film_14 = nn.Linear(n_w, base_channels * 2)
            self.film_28 = nn.Linear(n_w, (base_channels // 2) * 2)
            for fl in [self.film_7, self.film_14, self.film_28]:
                nn.init.zeros_(fl.weight)
                ch = fl.out_features // 2
                fl.bias.data[:ch] = 1.0   # γ = 1
                fl.bias.data[ch:] = 0.0   # β = 0

        if self.likelihood == "gaussian":
            self.logvar = nn.Parameter(torch.tensor(0.0))

    def _apply_film(self, film_layer, w, h):
        params = film_layer(w)                # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)  # each (B, C)
        # mask: 1 for cond=1 samples (w≠0), 0 for cond=0 samples (w=0)
        m = (w.abs().sum(dim=1, keepdim=True) > 0).float()  # (B, 1)
        gamma = m * gamma + (1.0 - m)        # cond=0 → γ=1 (identity)
        beta  = m * beta                     # cond=0 → β=0 (identity)
        gamma = gamma[:, :, None, None]       # (B, C, 1, 1)
        beta  = beta[:, :, None, None]
        return gamma * h + beta
        

    def forward(self, latent: torch.Tensor):
        B = latent.size(0)

        if self.film:
            w = latent[:, -self.n_w:]

        h = F.relu(self.fc1(latent))
        h = self.drop_fc(h)
        h = self.fc2(h).view(B, -1, 7, 7)  # (B, ch0, 7, 7)

        if self.film:
            h = self._apply_film(self.film_7, w, h)

        h = self.up1(h)
        h = F.relu(self.conv1a(h))
        h = F.relu(self.conv1b(h))

        if self.film:
            h = self._apply_film(self.film_14, w, h)

        h = self.up2(h)
        h = F.relu(self.conv2a(h))
        h = F.relu(self.conv2b(h))

        if self.film:
            h = self._apply_film(self.film_28, w, h)

        out = self.conv_out(h)  # logits (bernoulli) or mean (gaussian)

        if self.likelihood == "bernoulli":
            return out
        return out, self.logvar

    def negative_log_likelihood(self, y: torch.Tensor, *out_params, **kwargs):

        if self.likelihood == "bernoulli":
            logits = out_params[0]
            bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            # sum pixels, mean batch
            return bce.view(y.size(0), -1).sum(dim=1).mean()

        mu, logvar = out_params
        var = torch.exp(logvar)
        LOG2PI = math.log(2.0 * math.pi)
        nll = 0.5 * (LOG2PI + logvar + (y - mu) ** 2 / var)
        return nll.view(y.size(0), -1).sum(dim=1).mean()


