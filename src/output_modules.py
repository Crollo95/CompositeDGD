# src/output_modules.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseOutputModule(nn.Module):
    """
    Base class for decoder output heads.
    All subclasses must implement:
      - forward(h): returns parameters of the distribution
      - loss(y, *params): returns mean NLL over batch
    """

    def forward(self, h: torch.Tensor):
        raise NotImplementedError

    def loss(self, y: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def component_losses(self, y: torch.Tensor, *params: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Default: a single scalar loss for this module.
        Returns a dict mapping a name to a scalar NLL tensor.
        """
        nll = self.loss(y, *params)
        name = self.__class__.__name__
        return {name: nll}



class GaussianOutput(BaseOutputModule):
    """
    Diagonal Gaussian output:
      y | h ~ N(mean(h), diag(exp(log_var))).

    options:
      - train_log_var: bool (default True)
      - log_var_init: float (default 0.0)
    """

    def __init__(self, in_dim: int, out_dim: int, options: dict | None = None):
        super().__init__()
        if options is None:
            options = {}

        self.train_log_var = bool(options.get("train_log_var", True))
        log_var_init = float(options.get("log_var_init", 0.0))

        self.mean_layer = nn.Linear(in_dim, out_dim)

        init = torch.full((out_dim,), log_var_init)
        if self.train_log_var:
            self.log_var = nn.Parameter(init)
        else:
            self.register_buffer("log_var", init)

    def forward(self, h: torch.Tensor):
        mean = self.mean_layer(h)  # (B, out_dim)
        return mean, self.log_var

    def loss(self, y: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor):
        """
        Gaussian NLL: mean over batch.
        y: (B, F)
        mean: (B, F)
        log_var: (F,)
        """
        var = torch.exp(log_var)         # (F,)
        diff2 = (y - mean) ** 2          # (B, F)
        F_dim = y.shape[1]

        nll = 0.5 * (
            log_var.sum()
            + (diff2 / var).sum(dim=1)
            + F_dim * math.log(2.0 * math.pi)
        )  # (B,)
        return nll.mean()


class PoissonOutput(BaseOutputModule):
    """
    Poisson likelihood for non-negative counts:
      y | h ~ Poisson( rate(h) )
    """

    def __init__(self, in_dim: int, out_dim: int, options: dict | None = None):
        super().__init__()
        # could use options later (e.g. log1p transform flags)
        self.rate_layer = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor):
        # rate must be positive; use exp or softplus
        log_rate = self.rate_layer(h)
        rate = torch.exp(log_rate)  # (B, out_dim)
        return rate

    def loss(self, y: torch.Tensor, rate: torch.Tensor):
        """
        Poisson NLL: mean over batch.
        y: (B, F), rate: (B, F)
        NLL_i = sum_j [ rate_ij - y_ij * log(rate_ij) + log(y_ij!) ]
        log(y_ij!) is constant wrt parameters so we drop it.
        """
        eps = 1e-8
        nll = rate - y * torch.log(rate + eps)  # (B, F)
        return nll.sum(dim=1).mean()


class BernoulliOutput(BaseOutputModule):
    """
    Bernoulli output for binary features (0/1):
      y | h ~ Bernoulli(p(h))

    Suitable for columns that are strictly 0/1.
    """

    def __init__(self, in_dim: int, out_dim: int, options: dict | None = None):
        super().__init__()
        self.logit_layer = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor):
        """
        Returns logits for p(y=1).
        """
        logits = self.logit_layer(h)  # (B, out_dim)
        return logits

    def loss(self, y: torch.Tensor, logits: torch.Tensor):
        """
        Bernoulli NLL: mean over batch.
        y: (B, F) with values in {0,1}
        logits: (B, F)
        """
        # BCE with logits is numerically stable:
        # nll = - [ y*log(sigmoid) + (1-y)*log(1-sigmoid) ]
        nll = F.binary_cross_entropy_with_logits(logits, y, reduction="none")  # (B, F)
        return nll.sum(dim=1).mean()



class BetaOutput(BaseOutputModule):
    """
    Beta likelihood for features in (0,1):
      y | h ~ Beta(alpha(h), beta(h))

    Notes:
      - Beta is defined on (0,1) only. We clamp y to [eps, 1-eps].
      - Use this for proportions / rates, not for binary (use Bernoulli) or unbounded real (use Gaussian).

    options:
      - eps: float, clamp amount for y (default 1e-5)
      - min_concentration: float, lower bound added to concentration (default 2.0)
    """

    def __init__(self, in_dim: int, out_dim: int, options: dict | None = None):
        super().__init__()
        if options is None:
            options = {}

        self.eps = float(options.get("eps", 1e-5))
        self.min_concentration = float(options.get("min_concentration", 2.0))

        # Parameterization via mean + concentration (more stable than separate alpha/beta)
        self.mean_layer = nn.Linear(in_dim, out_dim)          # -> mean in (0,1) via sigmoid
        self.conc_layer = nn.Linear(in_dim, out_dim)          # -> concentration > 0 via softplus

    def forward(self, h: torch.Tensor):
        # mean in (0,1)
        m = torch.sigmoid(self.mean_layer(h))                 # (B, F)
        # concentration kappa > 0
        kappa = F.softplus(self.conc_layer(h)) + self.min_concentration  # (B, F)

        alpha = m * kappa
        beta = (1.0 - m) * kappa
        return alpha, beta

    def loss(self, y: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        """
        Beta NLL: mean over batch.
        y: (B, F) expected in [0,1] (we clamp to (0,1))
        """
        y = y.clamp(self.eps, 1.0 - self.eps)

        # log Beta pdf:
        # log p(y) = lgamma(a+b) - lgamma(a) - lgamma(b) + (a-1)log y + (b-1)log(1-y)
        lgamma = torch.lgamma
        logp = (
            lgamma(alpha + beta)
            - lgamma(alpha)
            - lgamma(beta)
            + (alpha - 1.0) * torch.log(y)
            + (beta - 1.0) * torch.log(1.0 - y)
        )  # (B, F)

        nll = -logp.sum(dim=1).mean()
        return nll



class NegativeBinomialOutput(BaseOutputModule):
    """
    Negative Binomial output for overdispersed counts:
      y | h ~ NB(mean=mu(h), dispersion=r)

    Options:
      - log_r_init: float (initial log-dispersion), default=0.0
      - learn_dispersion: bool, default=True
      - shared_dispersion: bool, default=False
          if True: one r shared across all features
          if False: per-feature r (vector of length out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, options: dict | None = None):
        super().__init__()
        if options is None:
            options = {}

        log_r_init = float(options.get("log_r_init", 0.0))
        learn_dispersion = bool(options.get("learn_dispersion", True))
        shared_dispersion = bool(options.get("shared_dispersion", False))

        self.mean_layer = nn.Linear(in_dim, out_dim)

        if shared_dispersion:
            init = torch.full((1,), log_r_init)
        else:
            init = torch.full((out_dim,), log_r_init)

        if learn_dispersion:
            self.log_r = nn.Parameter(init)
        else:
            self.register_buffer("log_r", init)

        self.shared_dispersion = shared_dispersion

    def forward(self, h: torch.Tensor):
        """
        Returns (mean, log_r):
          mean: (B, F)
          log_r: (F,) or (1,)
        """
        # mean must be positive; use softplus to avoid 0.
        mean = F.softplus(self.mean_layer(h))  # (B, out_dim)
        return mean, self.log_r

    def loss(self, y: torch.Tensor, mean: torch.Tensor, log_r: torch.Tensor):
        """
        Negative Binomial NLL: mean over batch.
        y: (B, F) non-negative counts
        mean: (B, F)
        log_r: (F,) or (1,)
        """
        eps = 1e-8
        # Ensure shapes: broadcast log_r over batch dimension
        if log_r.ndim == 1:
            # (F,) or (1,) -> (1, F)
            log_r = log_r.unsqueeze(0)  # (1, F)
        # Broadcast to (B, F)
        log_r = log_r.expand_as(mean)
        r = torch.exp(log_r)  # dispersion > 0

        # log NB pmf
        # log NB(y | mu, r) =
        #   lgamma(y + r) - lgamma(r) - lgamma(y+1)
        #   + r * (log r - log(r + mu))
        #   + y * (log mu - log(r + mu))
        lgamma = torch.lgamma

        log_mu = torch.log(mean + eps)
        log_r_plus_mu = torch.log(r + mean + eps)

        log_prob = (
            lgamma(y + r)
            - lgamma(r)
            - lgamma(y + 1.0)
            + r * (log_r - log_r_plus_mu)
            + y * (log_mu - log_r_plus_mu)
        )  # (B, F)

        nll = -log_prob.sum(dim=1).mean()
        return nll


class MultinomialOutput(BaseOutputModule):
    """
    Multinomial likelihood for a count vector over categories (features):

      y_i (vector) ~ Multinomial(N_i, p_i)
      where N_i = sum_j y_ij, p_i = softmax(logits_i)

    We drop the combinatorial constant term log(N!/prod y_j!) since it does not
    depend on model parameters (and thus does not affect gradients).

    Use this for groups of columns that form a compositional count vector.
    """

    def __init__(self, in_dim: int, out_dim: int, options: dict | None = None):
        super().__init__()
        self.logits_layer = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor):
        logits = self.logits_layer(h)  # (B, F_group)
        return logits

    def loss(self, y: torch.Tensor, logits: torch.Tensor):
        """
        Negative log-likelihood (up to an additive constant):
          NLL = -sum_j y_j * log p_j
        where p = softmax(logits).

        y: (B, F_group) expected to be nonnegative counts (can be floats, but "count-like")
        logits: (B, F_group)
        """
        # log p_j
        log_p = F.log_softmax(logits, dim=1)  # (B, F_group)

        # (B,) mean over batch; sum over categories
        nll = -(y * log_p).sum(dim=1).mean()
        return nll


class SurvivalCPHOutput(BaseOutputModule):
    """
    Cox Proportional Hazards (partial likelihood) head.

    Expected target y_sub has TWO columns:
      - time:  observed time (float)
      - event: 1 if event occurred, 0 if censored

    The head predicts a scalar risk score s_i (log hazard up to constant).

    Loss (negative partial log-likelihood, Breslow-style for ties via sorting trick):
      L = - sum_{i:event=1} [ s_i - log sum_{j: t_j >= t_i} exp(s_j) ]
    Normalized by number of events (default).

    Notes:
      - This is computed within each mini-batch, so it is an approximation if you use batches.
        For more faithful Cox training, use large batches (even full batch) for survival heads.
    """

    def __init__(self, in_dim: int, out_dim: int, options: dict | None = None):
        super().__init__()
        if out_dim != 2:
            raise ValueError(
                f"SurvivalCPHOutput expects out_dim=2 (time,event), got out_dim={out_dim}."
            )

        if options is None:
            options = {}

        self.time_index = int(options.get("time_index", 0))    # within y_sub
        self.event_index = int(options.get("event_index", 1))  # within y_sub

        # normalize by "events" (default) or "batch"
        self.normalize = str(options.get("normalize", "events")).lower()
        if self.normalize not in {"events", "batch"}:
            raise ValueError("SurvivalCPHOutput: normalize must be 'events' or 'batch'.")

        # optionally ignore rows with NaNs in time/event
        self.use_nan_mask = bool(options.get("use_nan_mask", True))

        # Predict scalar risk score
        self.risk_layer = nn.Linear(in_dim, 1)

    def forward(self, h: torch.Tensor):
        # (B, 1)
        return self.risk_layer(h)

    def loss(self, y: torch.Tensor, risk: torch.Tensor) -> torch.Tensor:
        """
        y:    (B, 2) [time, event]
        risk: (B, 1) or (B,)
        """
        # unpack
        time = y[:, self.time_index]
        event = y[:, self.event_index]

        # be robust to float event coding
        event = (event > 0.5).float()

        # optional NaN masking
        if self.use_nan_mask:
            valid = torch.isfinite(time) & torch.isfinite(event)
            time = time[valid]
            event = event[valid]
            risk = risk[valid]

        # handle empty
        if time.numel() == 0:
            return torch.zeros((), device=y.device, dtype=y.dtype)

        risk = risk.squeeze(-1)  # (B,)

        # sort by descending time so risk sets are prefixes
        order = torch.argsort(time, descending=True)
        risk_s = risk[order]
        event_s = event[order]

        # denom_i = log sum_{j<=i} exp(risk_s[j])  (prefix)
        log_denom = torch.logcumsumexp(risk_s, dim=0)

        # partial log-lik contributions only where event=1
        pll = (risk_s - log_denom) * event_s
        n_events = event_s.sum()

        if n_events.item() <= 0:
            # no events in this batch -> no Cox signal
            return torch.zeros((), device=y.device, dtype=y.dtype)

        nll = -pll.sum()

        if self.normalize == "events":
            nll = nll / n_events.clamp_min(1.0)
        else:  # "batch"
            nll = nll / max(1.0, float(event_s.shape[0]))

        return nll

    # ------------------- Breslow baseline + prediction ------------------- #

    def fit_breslow(self, time, event, risk):
        """
        Fit a Breslow baseline model from training data.

        time:  (N,) array-like
        event: (N,) array-like (0/1 or bool)
        risk:  (N,) array-like (linear predictors = risk scores)

        Stores:
          self.breslow_
          self.baseline_times_
          self.baseline_survival_
          self.baseline_survival_interp_
        """
        from sksurv.linear_model.coxph import BreslowEstimator
        from scipy.interpolate import interp1d
        import numpy as np

        # to numpy
        if torch.is_tensor(time):  time = time.detach().cpu().numpy()
        if torch.is_tensor(event): event = event.detach().cpu().numpy()
        if torch.is_tensor(risk):  risk = risk.detach().cpu().numpy()

        time = np.asarray(time, dtype=float).reshape(-1)
        event = np.asarray(event).reshape(-1).astype(bool)
        risk = np.asarray(risk, dtype=float).reshape(-1)

        # Fit Breslow
        self.breslow_ = BreslowEstimator().fit(risk, event, time)  # 

        # Baseline survival is the survival curve at risk=0 (exp(0)=1)
        baseline_sf = self.breslow_.get_survival_function(np.zeros(1, dtype=float))[0]  # 
        t0 = np.asarray(baseline_sf.x, dtype=float)
        s0 = np.asarray(baseline_sf(baseline_sf.x), dtype=float)

        # Ensure it starts at t=0 with S0(0)=1 for nice interpolation
        if t0.size == 0 or t0[0] > 0.0:
            t0 = np.concatenate([[0.0], t0])
            s0 = np.concatenate([[1.0], s0])

        self.baseline_times_ = t0
        self.baseline_survival_ = s0

        # step-wise interpolation (Cox baseline survival is a step function)
        self.baseline_survival_interp_ = interp1d(
            t0,
            s0,
            kind="previous",
            bounds_error=False,
            fill_value=(1.0, float(s0[-1])),
            assume_sorted=True,
        )

        return self

    def predict_survival_array(self, risk, times=None):
        """
        Predict survival probabilities on a time grid.

        risk:  (N,) array-like of risk scores (linear predictors)
        times: (T,) array-like; if None uses self.baseline_times_

        Returns:
          times (T,), survival (N,T)
        """
        import numpy as np

        if not hasattr(self, "baseline_survival_interp_"):
            raise RuntimeError("Call fit_breslow(...) before predict_survival_array(...).")

        if torch.is_tensor(risk):
            risk = risk.detach().cpu().numpy()

        risk = np.asarray(risk, dtype=float).reshape(-1)
        if times is None:
            times = self.baseline_times_
        times = np.asarray(times, dtype=float).reshape(-1)

        # baseline S0(t)
        s0_t = self.baseline_survival_interp_(times)  # (T,)
        s0_t = np.clip(s0_t, 1e-12, 1.0)

        # individual S(t|x) = S0(t) ^ exp(risk)
        hr = np.exp(risk)[:, None]          # (N,1)
        surv = np.power(s0_t[None, :], hr)  # (N,T)

        return times, surv

    def predict_survival_function(self, risk, times=None):
        """
        Same as predict_survival_array, but returns a callable per sample:
          f_i(t) -> S(t|x_i)

        Useful if you want StepFunction-like behavior without depending on sksurv objects.
        """
        from scipy.interpolate import interp1d
        import numpy as np

        times, surv = self.predict_survival_array(risk, times=times)

        funcs = []
        for i in range(surv.shape[0]):
            fi = interp1d(
                times,
                surv[i],
                kind="previous",
                bounds_error=False,
                fill_value=(1.0, float(surv[i, -1])),
                assume_sorted=True,
            )
            funcs.append(fi)
        return funcs



class CompositeOutput(BaseOutputModule):
    """
    Composite output: mix multiple heads over different feature subsets.

    options must contain:
      "heads": [
        {
          "module": "poisson" or "gaussian" or ...,
          "cols": [int, int, ...],             # feature indices for this head
          "options": { ... }                   # optional per-head options
        },
        ...
      ]
    """

    def __init__(self, in_dim: int, out_dim: int, options: dict):
        super().__init__()

        if options is None or "heads" not in options:
            raise ValueError(
                "CompositeOutput requires options with key 'heads', "
                "each head having 'module' and 'cols'."
            )

        heads_config = options["heads"]
        self.out_dim = out_dim

        # Validate configuration of heads and columns
        self._validate_heads_config(heads_config, out_dim)

        self.heads = nn.ModuleList()
        self.slices: list[list[int]] = []

        from .decoder import OUTPUT_MODULES  # local import to avoid cycles

        for head_cfg in heads_config:
            module_name = head_cfg["module"]
            cols = head_cfg["cols"]
            head_opts = head_cfg.get("options", None)

            if module_name not in OUTPUT_MODULES:
                raise ValueError(f"Unknown module '{module_name}' in CompositeOutput")

            self.slices.append(cols)

            HeadCls = OUTPUT_MODULES[module_name]
            # some modules accept 'options', some don't
            try:
                head = HeadCls(in_dim, len(cols), options=head_opts)
            except TypeError:
                head = HeadCls(in_dim, len(cols))
            self.heads.append(head)

    # --- tiny helper ------------------------------------------------- #
    @staticmethod
    def _validate_heads_config(heads_config: list[dict], out_dim: int) -> None:
        """
        Sanity-check the feature indices specified in heads_config.

        - All cols must be integers.
        - All cols must satisfy 0 <= col < out_dim.
        - No column index may appear in more than one head.
        """
        all_cols = []
        for i, head_cfg in enumerate(heads_config):
            if "cols" not in head_cfg:
                raise ValueError(f"Head {i} in CompositeOutput is missing 'cols' list.")
            cols = head_cfg["cols"]
            if not isinstance(cols, (list, tuple)):
                raise ValueError(f"Head {i} 'cols' must be a list/tuple of ints.")
            for c in cols:
                if not isinstance(c, int):
                    raise TypeError(f"Head {i} has non-integer column index: {c!r}")
                if c < 0 or c >= out_dim:
                    raise ValueError(
                        f"Head {i} has column index {c}, but valid range is [0, {out_dim})."
                    )
            all_cols.extend(cols)

        seen = set()
        dupes = set()
        for c in all_cols:
            if c in seen:
                dupes.add(c)
            else:
                seen.add(c)

        if dupes:
            dupes_sorted = sorted(dupes)
            raise ValueError(
                f"CompositeOutput: column indices appear in multiple heads: {dupes_sorted}. "
                "Each feature should belong to at most one head."
            )

    # ----------------------------------------------------------------- #

    def forward(self, h: torch.Tensor):
        """
        Returns a list of per-head outputs:
          [head0_out, head1_out, ...]
        where each head*_out is whatever that head's forward() returns
        (e.g. tensor, or (mean, log_var) tuple).
        """
        outputs = []
        for head in self.heads:
            outputs.append(head(h))
        return outputs

    def loss(self, y: torch.Tensor, *head_outputs):
        """
        Total NLL = sum of NLL from each head.
        y: (B, F)
        head_outputs: sequence of outputs from each head.forward(h)
        """
        if len(head_outputs) != len(self.heads):
            raise ValueError(
                f"CompositeOutput.loss expected {len(self.heads)} outputs, "
                f"got {len(head_outputs)}."
            )

        total_nll = 0.0
        for cols, head, out in zip(self.slices, self.heads, head_outputs):
            y_sub = y[:, cols]  # (B, len(cols))

            # out can be a tuple (mean, log_var) or a single tensor (rate, logits, ...)
            if isinstance(out, tuple):
                nll = head.loss(y_sub, *out)
            else:
                nll = head.loss(y_sub, out)

            total_nll = total_nll + nll

        return total_nll


    def component_losses(self, y: torch.Tensor, *head_outputs) -> dict[str, torch.Tensor]:
        """
        Return per-head mean NLLs as a dict:
          {"0_PoissonOutput": scalar, "1_GaussianOutput": scalar, ...}
        """
        if len(head_outputs) != len(self.heads):
            raise ValueError(
                f"CompositeOutput.component_losses expected {len(self.heads)} outputs, "
                f"got {len(head_outputs)}."
            )

        losses: dict[str, torch.Tensor] = {}
        for idx, (cols, head, out) in enumerate(zip(self.slices, self.heads, head_outputs)):
            y_sub = y[:, cols]  # (B, len(cols))

            if isinstance(out, tuple):
                nll = head.loss(y_sub, *out)
            else:
                nll = head.loss(y_sub, out)

            key = f"{idx}_{head.__class__.__name__}"
            losses[key] = nll

        return losses



