# Technical Overview of GeneralDGD

Technical breakdown of the model architecture, loss function, and training dynamics.

---

## 1. Model Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DGDModel                                    │
├─────────────────────────────────────────────────────────────────────┤
│  SplitRepresentationLayer (per-sample embeddings)                   │
│    ├─ z ∈ ℝ^{N × n_z}  (always active)                              │
│    └─ w ∈ ℝ^{N × n_w}  (masked by condition)                        │
│                                                                     │
│  GaussianMixturePrior (×2: one for z, one for w)                    │
│    ├─ means    ∈ ℝ^{K × D}    (learnable component centers)         │
│    ├─ log_vars ∈ ℝ^{K × D}    (learnable per-component variance)    │
│    └─ logits   ∈ ℝ^{K}        (mixture weights via softmax)         │
│                                                                     │
│  Decoder (MLP or CNN)                                               │
│    └─ OutputModule (Gaussian, Poisson, NB, Bernoulli, etc.)         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Split Latent Representation

The core innovation is the **split representation layer** (`latent.py:329-357`):

```python
class SplitRepresentationLayer(nn.Module):
    def __init__(self, n_samples, n_z, n_w):
        self.z = nn.Parameter(randn(n_samples, n_z) * 0.01)  # always learned
        self.w = nn.Parameter(randn(n_samples, n_w) * 0.01)  # conditional

    def forward(self, indices, cond_mask):
        z = self.z[indices]          # (B, n_z)
        w = self.w[indices]          # (B, n_w)
        w_used = w * cond_mask       # Zero when condition=0
        return z, w_used
```

**Key insight**: There is **no encoder**. Instead, each sample has its own learnable latent vector stored in a lookup table. This is a variational EM-style approach where:
- The decoder/priors are the "M-step" (model parameters)
- The representation layer is the "E-step" (per-sample posteriors)

**Conditioning mechanism**: When `cond_mask=0`, the w component is zeroed out, forcing the model to explain the data using only z. This enforces disentanglement.

---

## 3. Gaussian Mixture Prior

Each latent space (z and w) has its own GMM prior (`latent.py:68-305`):

### 3.1 GMM Parameters (Learnable)

| Parameter | Shape | Constraint | Prior |
|-----------|-------|------------|-------|
| `means` | (K, D) | None | **Softball** prior (soft ball boundary) |
| `log_vars` | (K, D) | None | **Gaussian** N(0, 1) |
| `logits` | (K,) | Softmax → π | **Dirichlet**(α=1) |

### 3.2 Softball Prior on Means (`latent.py:9-40`)

A novel "almost uniform inside a ball" distribution:

$$\log p(\boldsymbol{\mu}) = \log V - \log\left(1 + \exp\left(\text{sharpness}\cdot\left(\frac{\|\boldsymbol{\mu}\|}{\text{radius}} - 1\right)\right)\right)$$

- Inside the ball ($\|\boldsymbol{\mu}\| < \text{radius}$): nearly uniform
- Outside: exponential decay with steepness controlled by `sharpness`
- Default: `radius=5.0`, `sharpness=10.0`

### 3.3 Mixture Log-Probability

For a latent vector $\mathbf{z}$:

$$\log p(\mathbf{z} \mid \text{GMM}) = \log \sum_{k=1}^{K} \pi_k \; \mathcal{N}(\mathbf{z} \mid \boldsymbol{\mu}_k, \text{diag}(\boldsymbol{\sigma}^2_k))$$

Computed in `log_prob()` (`latent.py:185-193`):

```python
def log_prob(self, z):
    log_p_zk = self.log_prob_components(z)           # (B, K)
    log_weights = log_softmax(self.logits, dim=0)    # (K,)
    log_mix = log_p_zk + log_weights                 # (B, K)
    return logsumexp(log_mix, dim=1)                 # (B,)
```

---

## 4. Loss Function

The total loss (`model.py:79-201`) decomposes as:

$$\mathcal{L} = \underbrace{-\mathbb{E}_{i}\bigl[\log p(\mathbf{x}_i \mid \mathbf{z}_i, \mathbf{w}_i \odot c_i)\bigr]}_{\mathcal{L}_{\text{recon}}} + \beta_z \underbrace{\Bigl(-\mathbb{E}_{i}\bigl[\log p(\mathbf{z}_i \mid \text{GMM}_z)\bigr] - \tfrac{\log p(\boldsymbol{\theta}_z)}{n_0}\Bigr)}_{\mathcal{L}_{\text{prior},z}} + \beta_w \underbrace{\Bigl(-\mathbb{E}_{i \in \mathcal{B}_1}\bigl[\log p(\mathbf{w}_i \mid \text{GMM}_w)\bigr] - \tfrac{\log p(\boldsymbol{\theta}_w)}{n_1}\Bigr)}_{\mathcal{L}_{\text{prior},w}}$$

where $\mathbf{x}_i$ is the observed data, $c_i \in \{0,1\}$ the condition, $\beta_z = 1$ (fixed), $n_0 = |\mathcal{B}_0|$, $n_1 = |\mathcal{B}_1|$. The w prior is zero when $n_1 = 0$.

When `z_prior_on_y0_only=True`: the GMM$_z$ parameters $\boldsymbol{\theta}_z$ receive gradients only from cond=0 samples ($\mathcal{B}_0$), while z embeddings are pulled toward the GMM using all samples. This is implemented via a stop-gradient trick (see Section 4.2).

`batch_loss()` signature (`model.py:79`):

```python
def batch_loss(self, batch_y, rep_layer, batch_indices,
               cond_mask=None, z_prior_on_y0_only=False,
               scaling_factor=None):
```

- `scaling_factor`: optional `(batch,)` tensor of pre-computed library-size ratios.
  Required when using `NegativeBinomialOutput(mean_mode='softmax')`. If not provided,
  the method falls back to within-batch normalization (`sf = row_sum / median(batch_row_sums)`).

Returns an **8-tuple**:

```python
return (
    total,           # scalar: L_total
    recon_loss,      # scalar: reconstruction NLL
    prior_loss,      # scalar: β_z·L_prior_z + β_w·L_prior_w
    recon_components,# dict[str, scalar_tensor]
    prior_z_nll,     # scalar: -E[log p(z | GMM_z)]
    prior_z_params,  # scalar: -log p(θ_z) / batch_size
    prior_w_nll,     # scalar: -E[log p(w | GMM_w)]  (0 if no cond=1 samples)
    prior_w_params,  # scalar: -log p(θ_w) / batch_size  (0 if no cond=1 samples)
)
```

### 4.1 Reconstruction Loss

Depends on the output module:
- **Gaussian**: $\displaystyle\frac{1}{2}\sum_f \left[\log\sigma^2_f + \frac{(y_f - \mu_f)^2}{\sigma^2_f} + \log(2\pi)\right]$
- **Poisson**: $\displaystyle\sum_f \left[\lambda_f - y_f \log\lambda_f\right]$
- **Negative Binomial**: Full NB log-pmf with learnable dispersion $r$ (see Section 7.3)
- **Bernoulli**: $-\sum_f \left[y_f \log\sigma(l_f) + (1-y_f)\log(1-\sigma(l_f))\right]$

### 4.2 z Prior Loss (with conditional mode)

When `z_prior_on_y0_only=True` (`model.py:136-163`):

```python
# Split gradient flow:
# (A) z embeddings updated from ALL samples (frozen GMM params)
prior_z_nll_embed = self.prior_z.z_nll_detached_params(z)

# (B) GMM params updated only from y=0 samples (frozen z)
prior_z_nll_params = self.prior_z.z_nll(z_for_params.detach())

prior_z_nll = prior_z_nll_embed + prior_z_nll_params
```

This is clever: it means the GMM learns the structure of the **baseline** (condition=0) distribution, while z embeddings are still pulled toward that distribution for all samples.

### 4.3 w Prior Loss

Only computed for `condition=1` samples (`model.py:169-178`):

```python
active_mask = (cond_mask > 0.5)
if n_active > 0:
    w_active = w_used[active_mask]
    prior_w_nll = self.prior_w.z_nll(w_active)
    prior_w_params = self.prior_w.param_prior_loss(n_active)
```

---

## 5. Training Strategy

The training loop (`train.py:170-388`) uses a **multi-phase optimization** approach:

### Phase 1: Train (per epoch)
```
For each batch:
    1. Forward pass through rep_train → decoder
    2. Compute loss
    3. Update: prior_z, prior_w, decoder (immediate step)
    4. Accumulate gradients for rep_train
After all batches:
    5. Single step for rep_train
```

### Phase 2: Test/Validation (per epoch)
```
Freeze: model (priors + decoder)
For each batch:
    1. Forward pass through rep_test → decoder
    2. Accumulate gradients for rep_test only
After all batches:
    3. Single step for rep_test
Unfreeze: model
```

**Separate optimizers**:

| Component | Learning Rate | Optimizer |
|-----------|--------------|-----------|
| `prior_z`, `prior_w` | `lr_prior` (1e-3) | Adam |
| `decoder` | `lr_decoder` (1e-3) | Adam + weight_decay=1e-4 |
| `rep_train` | `lr_rep_train` (1e-2) | Adam |
| `rep_test` | `lr_rep_test` (1e-2) | Adam |

Note: Representations use 10× higher learning rate than model parameters.

### Beta-w Annealing (`train.py:191-199`)

When `beta_w_anneal=True`, the weight on the w prior is linearly warmed up from 0
to the target `beta_w` over the first 20% of epochs:

```python
warmup_epochs = max(1, int(0.2 * n_epochs))
model.beta_w = beta_w_target * min(1.0, epoch / warmup_epochs)
```

This prevents the w prior from dominating early training before w embeddings have
had a chance to learn meaningful structure. After training, `model.beta_w` is
restored to the original target value.

---

## 6. Decoder Architectures

### 6.1 MLP Decoder (`decoder.py:28-80`)

```
latent (n_z + n_w)
  → Linear(latent_dim, h[0]) + ReLU + Dropout
  → Linear(h[0], h[1]) + ReLU + Dropout
  → ...
  → OutputModule(h[-1], n_features)
```

### 6.2 CNN Decoder (`decoder.py:89-209`)

For 28×28 images. Selected automatically by `train_dgd()` when `X_train.ndim == 4`,
or explicitly via `decoder_type="cnn"` in `DGDModel`.

```
latent (n_z + n_w)
  → FC: Linear(latent_dim, 256) + ReLU + Dropout
  → FC: Linear(256, 128×7×7) → reshape (B, 128, 7, 7)
  → [FiLM @ 7×7]
  → Upsample 7→14: Conv(128→64) + ReLU + Conv(64→64) + ReLU
  → [FiLM @ 14×14]
  → Upsample 14→28: Conv(64→32) + ReLU + Conv(32→32) + ReLU
  → [FiLM @ 28×28]
  → Conv(32→C, 3×3)  # Output logits/mean
```

Uses bilinear upsampling. Output bias initialized to `logit(0.3)` for FashionMNIST.
Supports two likelihoods: `"bernoulli"` (returns logits) and `"gaussian"` (returns mean + learnable log_var).

### 6.3 FiLM Conditioning (`decoder.py:134-160`)

**Feature-wise Linear Modulation** (enabled via `film=True`, requires `n_w`).
At each resolution level, the w portion of the latent is projected to per-channel
scale (gamma) and shift (beta):

```python
def _apply_film(self, film_layer, w, h):
    gamma, beta = film_layer(w).chunk(2, dim=1)    # (B, C) each
    m = (w.abs().sum(dim=1, keepdim=True) > 0).float()  # 1 for cond=1, 0 for cond=0
    gamma = m * gamma + (1.0 - m)   # cond=0 → γ=1  (identity)
    beta  = m * beta                # cond=0 → β=0  (identity)
    return gamma[:,:,None,None] * h + beta[:,:,None,None]
```

- Initialized to identity: gamma bias=1, beta bias=0, weights=0.
- For cond=0 samples (w=0), FiLM reduces to identity (no effect on feature maps).
- Three FiLM layers: `film_7` (128 channels), `film_14` (64 channels), `film_28` (32 channels).

---

## 7. Output Modules (Likelihoods)

### 7.1 GaussianOutput (`output_modules.py:32-78`)

```python
# Parameters
mean = Linear(h)           # (B, F)
log_var = Parameter(F,)    # shared across samples, learnable
```

$$\mathcal{L} = \frac{1}{2}\sum_{f=1}^{F}\left[\log\sigma^2_f + \frac{(y_f - \mu_f)^2}{\sigma^2_f} + \log(2\pi)\right]$$

### 7.2 PoissonOutput (`output_modules.py:81-107`)

```python
rate = exp(Linear(h))      # (B, F), always positive
```

$$\mathcal{L} = \sum_{f=1}^{F}\left[\lambda_f - y_f \log\lambda_f\right] \quad \text{(ignoring the } \log\Gamma(y_f+1) \text{ constant)}$$

### 7.3 NegativeBinomialOutput (`output_modules.py:287-367`)

Two modes:
- **softplus**: `mean = softplus(Linear(h))` — absolute count prediction
- **softmax**: `mean = scaling_factor × softmax(Linear(h))` — library-size normalized

```python
# Dispersion (per-feature or shared)
r = exp(log_r) + dispersion_floor  # floor=1.0 by default
```

$$\log p(y \mid \mu, r) = \log\Gamma(y+r) - \log\Gamma(r) - \log\Gamma(y+1) + r\log\frac{r+\varepsilon}{r+\mu+\varepsilon} + y\log\frac{\mu+\varepsilon}{r+\mu+\varepsilon}$$

where $\varepsilon = 10^{-8}$ for numerical stability.

**Softmax mode** requires a `scaling_factor` argument (library-size ratio per sample).
In `train_dgd()`, global scaling factors are pre-computed once on the full training set
and reused for test (`train.py:60-72`):

```python
sf_train = row_sum_train / (median(row_sum_train) + 1e-8)
sf_test  = row_sum_test  / (median(row_sum_train) + 1e-8)   # same denominator
```

This ensures consistent normalization across train and test splits. If no global
scaling factor is provided to `batch_loss()`, it falls back to within-batch normalization.

### 7.4 BernoulliOutput (`output_modules.py:110-138`)

```python
logits = Linear(h)
```

$$\mathcal{L} = -\sum_{f=1}^{F}\left[y_f \log\sigma(l_f) + (1-y_f)\log(1-\sigma(l_f))\right]$$

### 7.5 BetaOutput (`output_modules.py:142-197`)

For proportions in (0,1):

```python
m = sigmoid(Linear(h))           # mean ∈ (0,1)
κ = softplus(Linear(h)) + 2.0    # concentration > 2
```

$$\alpha = m \cdot \kappa, \qquad \beta = (1-m) \cdot \kappa$$

$$\log p(y \mid \alpha, \beta) = \log\Gamma(\alpha+\beta) - \log\Gamma(\alpha) - \log\Gamma(\beta) + (\alpha-1)\log y + (\beta-1)\log(1-y)$$

### 7.6 MultinomialOutput (`output_modules.py:370-405`)

For compositional count vectors:

```python
logits = Linear(h)
log_p = log_softmax(logits, dim=1)
```

$$\mathcal{L} = -\frac{1}{B}\sum_{b=1}^{B}\sum_{f=1}^{F} y_{bf} \log p_{bf} \quad \text{(up to combinatorial constant)}$$

---

## 8. Key Equations Summary

### Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}_i \in \mathbb{R}^F$ | Observed feature vector for sample $i$ (`batch_y` in code) |
| $B$ | Batch size |
| $\mathcal{B}_0 = \{i : c_i = 0\}$, $n_0 = \lvert\mathcal{B}_0\rvert$ | Cond=0 (baseline) samples in batch |
| $\mathcal{B}_1 = \{i : c_i = 1\}$, $n_1 = \lvert\mathcal{B}_1\rvert$ | Cond=1 (active) samples in batch |

### Loss Function

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta_z\mathcal{L}_{\text{prior},z} + \beta_w\mathcal{L}_{\text{prior},w}$$

where $\beta_z = 1$ (fixed) and $\beta_w$ is a hyperparameter.

**Reconstruction** (all samples):

$$\mathcal{L}_{\text{recon}} = -\mathbb{E}_{i}\bigl[\log p\left(\mathbf{x}_i \mid \mathbf{z}_i, \mathbf{w}_i \odot c_i\right)\bigr]$$

**z prior** (all samples for embeddings; cond=0 only for GMM parameters):

$$\mathcal{L}_{\text{prior},z} = -\mathbb{E}_{i}\bigl[\log p(\mathbf{z}_i \mid \mathrm{GMM}_z)\bigr] - \frac{\log p(\boldsymbol{\theta}_z)}{n_0}$$

> **`z_prior_on_y0_only=True` (default):** z embeddings are pulled toward GMM$_z$ using all samples, but GMM$_z$ parameters $\boldsymbol{\theta}_z$ are fit only on $\mathcal{B}_0$ (cond=0). Implemented via two `.detach()` calls — gradient paths are disjoint. Without this flag, both collapse to a single NLL term over all samples with $n_0 = B$.

**w prior** (cond=1 samples only; zero when $n_1 = 0$):

$$\mathcal{L}_{\text{prior},w} = -\mathbb{E}_{i \in \mathcal{B}_1}\bigl[\log p(\mathbf{w}_i \mid \mathrm{GMM}_w)\bigr] - \frac{\log p(\boldsymbol{\theta}_w)}{n_1}$$

### GMM Log-Likelihood

$$\log p(\mathbf{z}\mid\text{GMM}) = \log\sum_{k=1}^{K} \pi_k \prod_{d=1}^{D} \mathcal{N}(z_d \mid \mu_{kd}, \sigma^2_{kd})$$

$$= \text{logsumexp}_k\left[\log\pi_k - \frac{D}{2}\log(2\pi) - \frac{1}{2}\sum_{d=1}^{D}\left(\log\sigma^2_{kd} + \frac{(z_d - \mu_{kd})^2}{\sigma^2_{kd}}\right)\right]$$

### Parameter Priors

$$\log p(\boldsymbol{\theta}) = \log p(\boldsymbol{\pi}\mid\text{Dir}(\alpha)) + \sum_k \log p(\boldsymbol{\mu}_k\mid\text{Softball}) + \sum_{k,d} \log p(\log\sigma^2_{kd}\mid\mathcal{N}(0,1))$$

### Conditional Masking

$$\text{latent input} = \left[\mathbf{z} \;\Big\|\; \mathbf{w} \odot c_{\text{mask}}\right], \qquad c_{\text{mask}} \in \{0, 1\}$$

---

## 9. Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_z` | 8 | Dimension of condition-independent latent |
| `n_w` | 8 | Dimension of condition-dependent latent |
| `n_components_z` | 8 | GMM components for z prior |
| `n_components_w` | 8 | GMM components for w prior |
| `beta_w` | 1.0 | Weight on w prior (can upweight/downweight) |
| `beta_w_anneal` | False | Warm up beta_w from 0 over the first 20% of epochs |
| `hidden_dims` | (64, 64) | MLP hidden layer sizes |
| `dropout_p` | 0.0 | Dropout probability |
| `z_prior_on_y0_only` | True | Train z GMM only on condition=0 samples |
| `decoder_type` | "mlp" | `"mlp"` for tabular data, `"cnn"` for 28x28 images (auto-selected by `train_dgd`) |
| `film` | False | Enable FiLM conditioning in DecoderCNN (requires `decoder_type="cnn"`) |
| `img_shape` | (1, 28, 28) | Image shape for CNN decoder (C, H, W) |
| `cnn_likelihood` | "bernoulli" | CNN output likelihood: `"bernoulli"` or `"gaussian"` |

### Evaluation parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `balanced_acc` | False | Use `balanced_accuracy_score` instead of `accuracy_score` for condition prediction classifiers. Useful with imbalanced classes (e.g. TCGA/GTEx). Accepted by `evaluate_model()`, threaded through `disentanglement_metrics()` → `cond_pred_train_test()` → `binary_metrics()`. |
