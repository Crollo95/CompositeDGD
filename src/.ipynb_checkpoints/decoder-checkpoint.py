# src/decoder.py
import torch
import torch.nn as nn

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

    def negative_log_likelihood(self, y: torch.Tensor, *out_params):
        """
        out_params is whatever forward() returned:
          - GaussianOutput: (mean, log_var)
          - PoissonOutput: rate
          - CompositeOutput: head_out0, head_out1, ...
        """
        return self.out.loss(y, *out_params)


