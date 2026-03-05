import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- Priors ----------------------------- #

class SoftballPrior(nn.Module):
    """
    "Softball" prior: almost uniform inside a ball of given radius,
    with a soft logistic boundary (like bulkDGD's softball prior).
    """

    def __init__(self, dim: int, radius: float, sharpness: float):
        super().__init__()
        self.dim = int(dim)
        self.radius = float(radius)
        self.sharpness = float(sharpness)

        # Normalization term for uniform distribution in a ball
        # (up to the soft boundary correction).
        self._log_vol = (
            math.lgamma(1.0 + 0.5 * self.dim)
            - self.dim * (math.log(self.radius) + 0.5 * math.log(math.pi))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., dim)
        returns: (...,) log-density
        """
        # radius norm
        r = x.norm(dim=-1) / self.radius
        # soft boundary via logistic; use log1p(exp()) for stability
        boundary = torch.log1p(torch.exp(self.sharpness * (r - 1.0)))
        return self._log_vol - boundary


class GaussianPrior(nn.Module):
    """
    Simple Gaussian prior over parameters (elementwise Normal).
    """

    def __init__(self, mean: float, stddev: float):
        super().__init__()
        self.mean = float(mean)
        self.stddev = float(stddev)
        self.var = self.stddev ** 2
        self._log_norm = -0.5 * math.log(2.0 * math.pi * self.var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: arbitrary shape
        returns: same shape log-density (elementwise)
        """
        return self._log_norm - 0.5 * ((x - self.mean) ** 2 / self.var)


# ---------------------- Gaussian Mixture Prior ------------------- #

class GaussianMixturePrior(nn.Module):
    """
    Gaussian mixture prior in latent space, with priors on:
      - component means (softball),
      - mixture weights (Dirichlet),
      - log-variances (Gaussian).
    """

    MEANS_PRIORS = ["softball"]
    WEIGHTS_PRIORS = ["dirichlet"]
    LOG_VAR_PRIORS = ["gaussian"]

    def __init__(
        self,
        latent_dim: int,
        n_components: int,
        means_prior_name: str = "softball",
        weights_prior_name: str = "dirichlet",
        log_var_prior_name: str = "gaussian",
        means_prior_options: dict | None = None,
        weights_prior_options: dict | None = None,
        log_var_prior_options: dict | None = None,
        init_means_std: float = 1.0,
        init_log_var: float = 0.0,
        init_logits_std: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_components = int(n_components)

        # GMM parameters (unconstrained) with non-symmetric init
        self.means = nn.Parameter(
            torch.randn(n_components, latent_dim) * init_means_std
        )
        self.log_vars = nn.Parameter(
            torch.zeros(n_components, latent_dim) + init_log_var
        )
        self.logits = nn.Parameter(
            torch.randn(n_components) * init_logits_std   # small random differences
        )

        # Store prior names
        self.means_prior_name = means_prior_name
        self.weights_prior_name = weights_prior_name
        self.log_var_prior_name = log_var_prior_name

        # Default options if not provided
        if means_prior_options is None:
            means_prior_options = {"radius": 5.0, "sharpness": 10.0}
        if weights_prior_options is None:
            weights_prior_options = {"alpha": 1.0}
        if log_var_prior_options is None:
            log_var_prior_options = {"mean": 0.0, "stddev": 1.0}

        self.means_prior_options = means_prior_options
        self.weights_prior_options = weights_prior_options
        self.log_var_prior_options = log_var_prior_options

        # Build prior distributions
        self._init_means_prior()
        self._init_log_var_prior()
        # weights prior is analytic; no nn.Module needed

    # ----------------------------- helpers ----------------------------- #

    @property
    def mixture_probs(self) -> torch.Tensor:
        """Softmax of logits → mixture weights (K,)."""
        return F.softmax(self.logits, dim=0)

    def _init_means_prior(self):
        if self.means_prior_name == "softball":
            radius = self.means_prior_options.get("radius", 5.0)
            sharpness = self.means_prior_options.get("sharpness", 10.0)
            self.means_prior_dist = SoftballPrior(
                dim=self.latent_dim, radius=radius, sharpness=sharpness
            )
        else:
            raise ValueError(
                f"Unsupported means_prior_name='{self.means_prior_name}'. "
                f"Supported: {self.MEANS_PRIORS}"
            )

    def _init_log_var_prior(self):
        if self.log_var_prior_name == "gaussian":
            mean = self.log_var_prior_options.get("mean", 0.0)
            stddev = self.log_var_prior_options.get("stddev", 1.0)
            self.log_var_prior_dist = GaussianPrior(mean=mean, stddev=stddev)
        else:
            raise ValueError(
                f"Unsupported log_var_prior_name='{self.log_var_prior_name}'. "
                f"Supported: {self.LOG_VAR_PRIORS}"
            )

    # --------------------- likelihood over z --------------------------- #

    def log_prob_components(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, latent_dim)
        returns: (batch, n_components) log p(z | component k)
        """
        B, D = z.shape
        z_exp = z.unsqueeze(1)                # (B,1,D)
        means = self.means.unsqueeze(0)       # (1,K,D)
        log_vars = self.log_vars.unsqueeze(0) # (1,K,D)

        inv_vars = torch.exp(-log_vars)
        diff2 = (z_exp - means) ** 2          # (B,K,D)

        log_prob = -0.5 * (
            D * math.log(2.0 * math.pi)
            + log_vars.sum(dim=-1)
            + (diff2 * inv_vars).sum(dim=-1)
        )  # (B,K)

        return log_prob

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Mixture log p(z): log ∑_k π_k N(z | μ_k, Σ_k).
        returns: (batch,)
        """
        log_p_zk = self.log_prob_components(z)           # (B,K)
        log_weights = F.log_softmax(self.logits, dim=0)  # (K,)
        log_mix = log_p_zk + log_weights.unsqueeze(0)    # (B,K)
        return torch.logsumexp(log_mix, dim=1)           # (B,)

    # -------------------- priors over parameters ----------------------- #

    def get_prior_log_prob(self) -> torch.Tensor:
        """
        Log p(parameters) = log p(weights) + log p(means) + log p(log_vars).
        Returns a scalar tensor.
        """
        logp = torch.tensor(0.0, device=self.means.device)

        # --- weights: Dirichlet(alpha) ---
        if self.weights_prior_name == "dirichlet":
            alpha = float(self.weights_prior_options.get("alpha", 1.0))
            # Dirichlet log-density: log C(α) + (α-1) * Σ_k log π_k
            pi = self.mixture_probs
            # Constant term (no gradient, but included for completeness)
            logC = (
                math.lgamma(alpha * self.n_components)
                - self.n_components * math.lgamma(alpha)
            )
            logp = logp + logC + (alpha - 1.0) * pi.log().sum()
        else:
            raise ValueError(
                f"Unsupported weights_prior_name='{self.weights_prior_name}'. "
                f"Supported: {self.WEIGHTS_PRIORS}"
            )

        # --- means: softball prior ---
        if self.means_prior_name == "softball":
            # means_prior_dist.log_prob returns (...,) over last dim
            logp = logp + self.means_prior_dist.log_prob(self.means).sum()
        else:
            raise ValueError(
                f"Unsupported means_prior_name='{self.means_prior_name}'. "
                f"Supported: {self.MEANS_PRIORS}"
            )

        # --- log-variances: Gaussian prior ---
        if self.log_var_prior_name == "gaussian":
            # GaussianPrior.log_prob returns elementwise log-density
            logp = logp + self.log_var_prior_dist.log_prob(self.log_vars).sum()
        else:
            raise ValueError(
                f"Unsupported log_var_prior_name='{self.log_var_prior_name}'. "
                f"Supported: {self.LOG_VAR_PRIORS}"
            )

        return logp

    # -------------------- separate loss pieces ------------------------- #

    def z_nll(self, z: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood of z under the mixture:
          - E_batch[ log p(z | GMM) ].
        Returns a scalar tensor.
        """
        log_p_z = self.log_prob(z)  # (B,)
        return -log_p_z.mean()

    def param_prior_loss(self, batch_size: int) -> torch.Tensor:
        """
        Negative log prior over GMM parameters (weights, means, log_vars),
        scaled by batch_size to keep magnitudes comparable to z_nll.
        """
        prior_logp = self.get_prior_log_prob()  # scalar log p(parameters)
        return -prior_logp / batch_size

    # --------------------------- loss ---------------------------------- #

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Total GMM loss:
          - average negative log p(z) under the mixture
          - minus log prior over parameters (means, log_vars, weights),
            scaled by number of samples to keep magnitudes reasonable.
        """
        nll_z = self.z_nll(z)
        prior_loss = self.param_prior_loss(z.shape[0])
        return nll_z + prior_loss


# --------------------- Representation Layer --------------------------- #

class RepresentationLayer(nn.Module):
    """
    bulkDGD-like representation layer:
    stores one latent vector z_i per sample as a learnable parameter.
    """

    def __init__(self, n_samples: int, latent_dim: int, init_std: float = 0.01):
        super().__init__()
        init = torch.randn(n_samples, latent_dim) * init_std
        self.values = nn.Parameter(init)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: (batch,) long tensor of sample indices
        returns: z for those indices, shape (batch, latent_dim)
        """
        return self.values[indices]
