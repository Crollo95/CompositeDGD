import numpy as np
import pandas as pd
import torch

from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.special import digamma

from .metrics_predictors import (
    DEFAULT_CLF_KIND,
    cond_pred_train_test,
    cluster_pred_train_test_mcc,
)
from .output_modules import NegativeBinomialOutput


# ================================================================
# Helpers
# ================================================================

def to_numpy(x):
    """Convert torch/pandas/numpy to numpy array safely."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if hasattr(x, "values"):  # pandas
        return x.values
    return np.asarray(x)


def as_2d(x):
    """Flatten everything except first axis."""
    x = to_numpy(x)
    return x.reshape(x.shape[0], -1)


def concat_X(Xa, Xb):
    """Concatenate along axis 0. Works for numpy arrays or torch tensors."""
    if torch.is_tensor(Xa) or torch.is_tensor(Xb):
        if not torch.is_tensor(Xa):
            Xa = torch.as_tensor(Xa)
        if not torch.is_tensor(Xb):
            Xb = torch.as_tensor(Xb)
        return torch.cat([Xa, Xb], dim=0)
    else:
        return np.concatenate([Xa, Xb], axis=0)


def infer_cluster_keys(ds):
    """Heuristic defaults for FashionMNIST overlay dataset."""
    z_key = "fashion_label" if "fashion_label_train" in ds else None
    w_key = "mnist_label" if "mnist_label_train" in ds else None
    return z_key, w_key


# ================================================================
# Latents + reconstruction
# ================================================================

@torch.no_grad()
def get_latents_split(rep_layer, cond, device, batch_size=2048):
    """Returns (z, w_used) as numpy arrays from a SplitRepresentationLayer."""
    rep_layer.eval()
    cond = np.asarray(cond).ravel()
    N = len(cond)

    idx_all = torch.arange(N, device=device, dtype=torch.long)
    cond_t = torch.as_tensor(cond, device=device, dtype=torch.float32)

    zs, ws = [], []
    for s in range(0, N, batch_size):
        b = idx_all[s:s+batch_size]
        cm = cond_t[s:s+batch_size]
        z, w_used = rep_layer(b, cond_mask=cm)
        zs.append(z.detach().cpu())
        ws.append(w_used.detach().cpu())

    return torch.cat(zs).numpy(), torch.cat(ws).numpy()


@torch.no_grad()
def recon_from_model(model, z, w_used, device, batch_size=2048):
    """
    Decode latents to data space.
    Bernoulli decoder -> sigmoid(logits); Gaussian decoder -> mu.
    """
    model.eval()
    z = np.asarray(z); w_used = np.asarray(w_used)
    N = z.shape[0]

    outs = []
    is_bern = getattr(model.decoder, "likelihood", "").lower() == "bernoulli"

    for s in range(0, N, batch_size):
        zt = torch.as_tensor(z[s:s+batch_size], device=device, dtype=torch.float32)
        wt = torch.as_tensor(w_used[s:s+batch_size], device=device, dtype=torch.float32)
        latent = torch.cat([zt, wt], dim=1)

        out = model.decoder(latent)
        logits_or_mu = out[0] if isinstance(out, (list, tuple)) else out

        xhat = torch.sigmoid(logits_or_mu) if is_bern else logits_or_mu
        outs.append(xhat.detach().cpu())

    return torch.cat(outs).numpy()


@torch.no_grad()
def recon_nll_from_model(model, X, z, w_used, device, batch_size=2048):
    """Average reconstruction NLL per sample."""
    model.eval()
    X = np.asarray(X); z = np.asarray(z); w_used = np.asarray(w_used)
    N = z.shape[0]
    total = 0.0

    for s in range(0, N, batch_size):
        xb = torch.as_tensor(X[s:s+batch_size], device=device, dtype=torch.float32)
        zt = torch.as_tensor(z[s:s+batch_size], device=device, dtype=torch.float32)
        wt = torch.as_tensor(w_used[s:s+batch_size], device=device, dtype=torch.float32)

        latent = torch.cat([zt, wt], dim=1)

        out = model.decoder(latent)
        out_params = out if isinstance(out, (list, tuple)) else (out,)

        scaling_factor = None
        if isinstance(getattr(model.decoder, "out", None), NegativeBinomialOutput):
            if model.decoder.out.mean_mode == "softmax":
                sf = xb.sum(dim=1)
                scaling_factor = sf / (sf.median() + 1e-8)

        nll = model.decoder.negative_log_likelihood(xb, *out_params, scaling_factor=scaling_factor)
        total += float(nll.item()) * xb.shape[0]

    return total / float(N)


# ================================================================
# Low-level metric helpers
# ================================================================

def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rmse_by_group(X, Xhat, y_binary):
    y = np.asarray(y_binary).ravel()
    m0, m1 = (y < 0.5), (y > 0.5)
    return {
        "all": rmse(X, Xhat),
        "y0": rmse(X[m0], Xhat[m0]) if np.any(m0) else np.nan,
        "y1": rmse(X[m1], Xhat[m1]) if np.any(m1) else np.nan,
    }


def mi_y_latent(y_binary, Z):
    y = np.asarray(y_binary).astype(int).ravel()
    Z = np.asarray(Z)
    return float(np.sum(mutual_info_classif(Z, y, discrete_features=False, random_state=0)))


def mi_ksg(X, Y, k=10):
    X = np.asarray(X); Y = np.asarray(Y)
    XY = np.concatenate([X, Y], axis=1)
    n = XY.shape[0]

    nn_xy = NearestNeighbors(metric="chebyshev", n_neighbors=k+1).fit(XY)
    dist, _ = nn_xy.kneighbors(XY)
    eps = dist[:, k]

    nn_x = NearestNeighbors(metric="chebyshev").fit(X)
    nn_y = NearestNeighbors(metric="chebyshev").fit(Y)

    nx = np.array([len(nn_x.radius_neighbors([X[i]], radius=eps[i]-1e-15, return_distance=False)[0]) - 1
                   for i in range(n)])
    ny = np.array([len(nn_y.radius_neighbors([Y[i]], radius=eps[i]-1e-15, return_distance=False)[0]) - 1
                   for i in range(n)])

    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def kmeans_nmi(X_embed, labels, n_clusters=None, seed=0, n_init=20):
    X_embed = np.asarray(X_embed)
    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit_predict(X_embed)
    return float(normalized_mutual_info_score(labels, pred))


def best_1d_nmi(X_embed, labels, n_clusters=None, seed=0, n_init=20):
    X_embed = np.asarray(X_embed)
    labels = np.asarray(labels)
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    best = -np.inf
    for j in range(X_embed.shape[1]):
        xj = X_embed[:, [j]]
        pred = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed).fit_predict(xj)
        best = max(best, normalized_mutual_info_score(labels, pred))
    return float(best)


# ================================================================
# Focused metric groups
# ================================================================

def recon_metrics(model, z, w_used, X, cond, device):
    """
    Reconstruction quality for one split.

    Returns (metrics_dict, cache_dict).
    Keys: nll_joint, nll_zonly, rmse_joint_{all,y0,y1}, rmse_zonly_{...},
          rmse_gain_{...}, delta_xhat_active.
    Cache: xhat_joint, xhat_zonly, X_2d, xhat_joint_2d, xhat_zonly_2d.
    """
    X_np = to_numpy(X)
    cond = np.asarray(cond).ravel()

    xhat_joint = recon_from_model(model, z, w_used, device=device)
    xhat_zonly = recon_from_model(model, z, np.zeros_like(w_used), device=device)

    X_2d = as_2d(X_np)
    xj_2d = as_2d(xhat_joint)
    xz_2d = as_2d(xhat_zonly)

    out = {}
    for k, v in rmse_by_group(X_2d, xj_2d, cond).items():
        out[f"rmse_joint_{k}"] = v
    for k, v in rmse_by_group(X_2d, xz_2d, cond).items():
        out[f"rmse_zonly_{k}"] = v

    out["rmse_gain_all"] = out["rmse_zonly_all"] - out["rmse_joint_all"]
    out["rmse_gain_y0"]  = out["rmse_zonly_y0"]  - out["rmse_joint_y0"]
    out["rmse_gain_y1"]  = out["rmse_zonly_y1"]  - out["rmse_joint_y1"]

    out["nll_joint"] = recon_nll_from_model(model, X_np, z, w_used, device=device)
    out["nll_zonly"] = recon_nll_from_model(model, X_np, z, np.zeros_like(w_used), device=device)

    active = cond > 0.5
    out["delta_xhat_active"] = float(np.mean(np.abs(xj_2d[active] - xz_2d[active]))) if np.any(active) else np.nan

    cache = {
        "xhat_joint": xhat_joint, "xhat_zonly": xhat_zonly,
        "X_2d": X_2d, "xhat_joint_2d": xj_2d, "xhat_zonly_2d": xz_2d,
    }
    return out, cache


def disentanglement_metrics(
    z_train, cond_train, cache_train,
    z_test, cond_test, cache_test,
    clf_kind=None, seed=0, balanced_acc=False,
):
    """
    Condition-invariance of z. Classifiers trained on train, evaluated on test.
    MI/NMI computed on test split only.

    cache_train/test: dicts from recon_metrics
        (need keys: X_2d, xhat_joint_2d, xhat_zonly_2d).

    Returns flat dict. Classifier test-side values are bare keys;
    train-side values get a _train suffix.
    """
    if clf_kind is None:
        clf_kind = DEFAULT_CLF_KIND

    out = {}

    def _add_clf(prefix, Xtr, Xte):
        rows = cond_pred_train_test(
            Xtr, cond_train, Xte, cond_test,
            seed=seed, prefix=prefix, kind=clf_kind, scale=True, balanced=balanced_acc,
        )
        for name, d in rows.items():
            out[name] = d["test"]
            out[f"{name}_train"] = d["train"]

    _add_clf("cond_from_Z", z_train, z_test)
    _add_clf("cond_from_X", cache_train["X_2d"], cache_test["X_2d"])
    _add_clf("cond_from_xhat_joint", cache_train["xhat_joint_2d"], cache_test["xhat_joint_2d"])
    _add_clf("cond_from_xhat_zonly", cache_train["xhat_zonly_2d"], cache_test["xhat_zonly_2d"])

    # Centered AUC
    auc = out.get("cond_from_Z_auc", np.nan)
    out["cond_from_Z_centered_auc"] = float(abs(float(auc) - 0.5)) if np.isfinite(auc) else np.nan

    # Info-theoretic (test split)
    out["I_y_z"] = mi_y_latent(cond_test, z_test)
    out["nmi_kmeans_y_from_z"] = kmeans_nmi(z_test, cond_test, n_clusters=2, seed=seed)
    out["nmi_best1d_y_from_z"] = best_1d_nmi(z_test, cond_test, n_clusters=2, seed=seed)

    return out


def utility_metrics(
    z_train, z_test,
    w_train, w_test,
    cond_train, cond_test,
    z_labels_train=None, z_labels_test=None,
    w_labels_train=None, w_labels_test=None,
    clf_kind=None, seed=0,
):
    """
    Representation usefulness:
      z -> cluster labels (MCC),  w -> condition-specific labels (MCC, active only).

    Returns flat dict with keys: z_cluster_mcc, w_cluster_mcc_active.
    """
    if clf_kind is None:
        clf_kind = DEFAULT_CLF_KIND

    out = {"z_cluster_mcc": 0.0, "w_cluster_mcc_active": 0.0}

    # Z -> cluster labels
    if z_labels_train is not None and z_labels_test is not None:
        zlab_tr = np.asarray(z_labels_train).ravel()
        zlab_te = np.asarray(z_labels_test).ravel()
        if len(np.unique(zlab_tr)) > 1 and len(np.unique(zlab_te)) > 1:
            out["z_cluster_mcc"] = float(cluster_pred_train_test_mcc(
                z_train, zlab_tr, z_test, zlab_te, seed=seed, kind=clf_kind,
            )["mcc"]["test"])

    # W -> condition-specific labels (active samples only)
    if w_labels_train is not None and w_labels_test is not None:
        wlab_tr = np.asarray(w_labels_train).ravel()
        wlab_te = np.asarray(w_labels_test).ravel()
        cond_tr = np.asarray(cond_train).ravel()
        cond_te = np.asarray(cond_test).ravel()

        m_tr = (cond_tr > 0.5) & (wlab_tr != -1)
        m_te = (cond_te > 0.5) & (wlab_te != -1)

        if (m_tr.sum() > 0 and m_te.sum() > 0
                and len(np.unique(wlab_tr[m_tr])) > 1
                and len(np.unique(wlab_te[m_te])) > 1):
            out["w_cluster_mcc_active"] = float(cluster_pred_train_test_mcc(
                w_train[m_tr], wlab_tr[m_tr],
                w_test[m_te], wlab_te[m_te],
                seed=seed, kind=clf_kind,
            )["mcc"]["test"])

    return out


# ================================================================
# Unified entry point
# ================================================================

def evaluate_model(
    model, ds_bundle, device,
    clf_kind=None, seed=0,
    z_cluster_key=None, w_cluster_key=None,
    compute_ksg=False, ksg_subsample=3000, balanced_acc=False,
):
    """
    Full model evaluation.

    Returns flat dict with category-prefixed keys:
        recon/*         reconstruction quality (test split)
        disentangle/*   condition-invariance of z
        utility/*       representation usefulness

    ds_bundle must contain:
        X_train (or X_trainval), X_test
        cond_train (or cond_trainval), cond_test
    Optionally for utility metrics:
        {z_cluster_key}_{train|trainval}, {z_cluster_key}_test
        {w_cluster_key}_{train|trainval}, {w_cluster_key}_test
    """
    if clf_kind is None:
        clf_kind = DEFAULT_CLF_KIND
    ds = dict(ds_bundle)

    # --- determine train-side data ---
    if "X_trainval" in ds:
        X_tr = ds["X_trainval"]
        cond_tr = np.asarray(ds["cond_trainval"]).ravel()
        label_suffix_tr = "trainval"
    else:
        X_tr = ds["X_train"]
        cond_tr = np.asarray(ds["cond_train"]).ravel()
        label_suffix_tr = "train"

    X_te = ds["X_test"]
    cond_te = np.asarray(ds["cond_test"]).ravel()

    # --- latents ---
    z_tr, w_tr = get_latents_split(model.rep_train, cond_tr, device=device)
    z_te, w_te = get_latents_split(model.rep_test, cond_te, device=device)

    # --- reconstruction ---
    recon_te, cache_te = recon_metrics(model, z_te, w_te, X_te, cond_te, device)
    _, cache_tr = recon_metrics(model, z_tr, w_tr, X_tr, cond_tr, device)

    # --- disentanglement ---
    dis = disentanglement_metrics(
        z_tr, cond_tr, cache_tr,
        z_te, cond_te, cache_te,
        clf_kind=clf_kind, seed=seed, balanced_acc=balanced_acc,
    )

    # --- utility (cluster labels) ---
    if z_cluster_key is None or w_cluster_key is None:
        z_inferred, w_inferred = infer_cluster_keys(ds)
        if z_cluster_key is None:
            z_cluster_key = z_inferred
        if w_cluster_key is None:
            w_cluster_key = w_inferred

    z_labels_tr = z_labels_te = w_labels_tr = w_labels_te = None

    if z_cluster_key is not None:
        k_tr = f"{z_cluster_key}_{label_suffix_tr}"
        k_te = f"{z_cluster_key}_test"
        if k_tr in ds and k_te in ds:
            z_labels_tr = ds[k_tr]
            z_labels_te = ds[k_te]

    if w_cluster_key is not None:
        k_tr = f"{w_cluster_key}_{label_suffix_tr}"
        k_te = f"{w_cluster_key}_test"
        if k_tr in ds and k_te in ds:
            w_labels_tr = ds[k_tr]
            w_labels_te = ds[k_te]

    util = utility_metrics(
        z_tr, z_te, w_tr, w_te, cond_tr, cond_te,
        z_labels_train=z_labels_tr, z_labels_test=z_labels_te,
        w_labels_train=w_labels_tr, w_labels_test=w_labels_te,
        clf_kind=clf_kind, seed=seed,
    )

    # --- optional KSG MI ---
    ksg_val = np.nan
    if compute_ksg:
        rng = np.random.default_rng(seed)
        n = z_te.shape[0]
        m = min(n, int(ksg_subsample))
        idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
        ksg_val = mi_ksg(z_te[idx], w_te[idx], k=10)

    # --- assemble with category prefixes ---
    result = {}
    for k, v in recon_te.items():
        result[f"recon/{k}"] = v
    for k, v in dis.items():
        result[f"disentangle/{k}"] = v
    for k, v in util.items():
        result[f"utility/{k}"] = v
    result["disentangle/I_z_w_ksg"] = ksg_val

    return result


# ================================================================
# Convenience wrapper (notebook backward compat)
# ================================================================

def build_df_metrics(
    model, X_train, cond_train, X_test, cond_test, device,
    clf_kind=None, seed=0, **extra_bundle_keys,
):
    """
    Convenience wrapper around evaluate_model for notebook use.
    Returns a DataFrame with metric names as index.
    """
    ds = {
        "X_train": X_train, "X_test": X_test,
        "cond_train": cond_train, "cond_test": cond_test,
    }
    ds.update(extra_bundle_keys)
    result = evaluate_model(model, ds, device, clf_kind=clf_kind, seed=seed)
    return pd.DataFrame.from_dict(result, orient="index", columns=["value"])
