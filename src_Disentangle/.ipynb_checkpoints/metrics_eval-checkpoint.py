import numpy as np
import pandas as pd
import torch

#from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.special import digamma

from .metrics_predictors import cond_pred_train_test, cluster_pred_train_test_mcc
from .output_modules import NegativeBinomialOutput

# --------------------------
# Small helpers
# --------------------------
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
    """Concatenate Xa and Xb along axis 0. Works for numpy arrays or torch tensors (e.g. images)."""
    if torch.is_tensor(Xa) or torch.is_tensor(Xb):
        if not torch.is_tensor(Xa):
            Xa = torch.as_tensor(Xa)
        if not torch.is_tensor(Xb):
            Xb = torch.as_tensor(Xb)
        return torch.cat([Xa, Xb], dim=0)
    else:
        return np.concatenate([Xa, Xb], axis=0)


# --------------------------
# Latents + recon
# --------------------------
@torch.no_grad()
def get_latents_split(rep_layer, cond, device, batch_size=2048):
    """
    Returns (z, w_used) from SplitRepresentationLayer aligned with cond.
    """
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
    Returns reconstruction in the 'data space':
      - Bernoulli decoder -> sigmoid(logits)
      - Gaussian decoder  -> mu
    Output shape matches decoder output (e.g. N x 1 x 28 x 28 for images).
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
    """
    Average reconstruction NLL per sample under the model decoder.

    Works for:
      - MLP decoder (Gaussian/Poisson/NB/...)
      - CNN decoder (Bernoulli/Gaussian)

    For NB(mean_mode='softmax'), uses the SAME scaling_factor logic as model.batch_loss:
      sf_norm = sf / median(sf) within the evaluation minibatch.
    """
    model.eval()
    X = np.asarray(X)
    z = np.asarray(z)
    w_used = np.asarray(w_used)

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
        # match model.batch_loss NB scaling logic
        if isinstance(getattr(model.decoder, "out", None), NegativeBinomialOutput):
            if model.decoder.out.mean_mode == "softmax":
                sf = xb.sum(dim=1)  # (B,)
                scaling_factor = sf / (sf.median() + 1e-8)

        nll = model.decoder.negative_log_likelihood(xb, *out_params, scaling_factor=scaling_factor)
        bs = xb.shape[0]
        total += float(nll.item()) * bs

    return total / float(N)


# --------------------------
# RMSE
# --------------------------
def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def rmse_by_group(X, Xhat, y_binary):
    y = np.asarray(y_binary).ravel()
    out = {}
    out["rmse_all"] = rmse(X, Xhat)

    m0 = (y < 0.5)
    m1 = (y > 0.5)
    out["rmse_y0"] = rmse(X[m0], Xhat[m0]) if np.any(m0) else np.nan
    out["rmse_y1"] = rmse(X[m1], Xhat[m1]) if np.any(m1) else np.nan
    return out


# --------------------------
# MI + NMI metrics (unchanged)
# --------------------------
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
        nmi = normalized_mutual_info_score(labels, pred)
        best = max(best, nmi)
    return float(best)


# --------------------------
# Main split eval
# --------------------------
def eval_split(
    model, rep_layer, X, cond, device,
    split_name="train",
    cell_type=None,
    compute_ksg=False,
    ksg_subsample=3000,
    seed=0,
    flatten_for_metrics=True,
):
    X = to_numpy(X)
    cond = np.asarray(cond).ravel()

    z, w_used = get_latents_split(rep_layer, cond, device=device)

    xhat_joint = recon_from_model(model, z, w_used, device=device)
    xhat_zonly = recon_from_model(model, z, np.zeros_like(w_used), device=device)

    out = {"split": split_name}

    # views for RMSE/logreg
    X_m  = as_2d(X) if flatten_for_metrics else X
    xj_m = as_2d(xhat_joint) if flatten_for_metrics else xhat_joint
    xz_m = as_2d(xhat_zonly) if flatten_for_metrics else xhat_zonly

    out.update({f"rmse_joint_{k}": v for k, v in rmse_by_group(X_m, xj_m, cond).items()})
    out.update({f"rmse_zonly_{k}": v for k, v in rmse_by_group(X_m, xz_m, cond).items()})

    out["rmse_gain_all"] = out["rmse_zonly_rmse_all"] - out["rmse_joint_rmse_all"]
    out["rmse_gain_y0"]  = out["rmse_zonly_rmse_y0"]  - out["rmse_joint_rmse_y0"]
    out["rmse_gain_y1"]  = out["rmse_zonly_rmse_y1"]  - out["rmse_joint_rmse_y1"]

    out["recon_nll_joint"] = recon_nll_from_model(model, X, z, w_used, device=device)
    out["recon_nll_zonly"] = recon_nll_from_model(model, X, z, np.zeros_like(w_used), device=device)

    active = cond > 0.5
    if np.any(active):
        out["mean_abs_delta_xhat_active"] = float(np.mean(np.abs(xj_m[active] - xz_m[active])))
    else:
        out["mean_abs_delta_xhat_active"] = np.nan

    out["I_y_z"] = mi_y_latent(cond, z)
    out["I_y_w"] = mi_y_latent(cond, w_used)

    if compute_ksg:
        rng = np.random.default_rng(seed)
        n = z.shape[0]
        m = min(n, int(ksg_subsample))
        idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)
        out["I_z_w_ksg"] = mi_ksg(z[idx], w_used[idx], k=10)
    else:
        out["I_z_w_ksg"] = np.nan

    out["nmi_kmeans_y_from_w"] = kmeans_nmi(w_used, cond, n_clusters=2, seed=seed)
    out["nmi_best1d_y_from_w"] = best_1d_nmi(w_used, cond, n_clusters=2, seed=seed)
    out["nmi_kmeans_y_from_z"] = kmeans_nmi(z, cond, n_clusters=2, seed=seed)
    out["nmi_best1d_y_from_z"] = best_1d_nmi(z, cond, n_clusters=2, seed=seed)

    if cell_type is not None:
        cell_type = np.asarray(cell_type)
        k_ct = len(np.unique(cell_type))
        out["nmi_kmeans_ct_from_z"] = kmeans_nmi(z, cell_type, n_clusters=k_ct, seed=seed)
        out["nmi_best1d_ct_from_z"] = best_1d_nmi(z, cell_type, n_clusters=k_ct, seed=seed)

        if np.any(active):
            out["nmi_kmeans_ct_from_w_active"] = kmeans_nmi(
                w_used[active], cell_type[active],
                n_clusters=len(np.unique(cell_type[active])), seed=seed
            )
            out["nmi_best1d_ct_from_w_active"] = best_1d_nmi(
                w_used[active], cell_type[active],
                n_clusters=len(np.unique(cell_type[active])), seed=seed
            )
        else:
            out["nmi_kmeans_ct_from_w_active"] = np.nan
            out["nmi_best1d_ct_from_w_active"] = np.nan

    cache = {
        "z": z,
        "w_used": w_used,
        "xhat_joint": xhat_joint,   # keeps original decoder shape (images ok)
        "xhat_zonly": xhat_zonly,
        "X": X,
        "X_2d": X_m,
        "xhat_joint_2d": xj_m,
        "xhat_zonly_2d": xz_m,
    }
    return out, cache


def build_df_metrics(
    model,
    X_train, cond_train,
    X_test,  cond_test,
    device,
    cell_types_train=None,
    cell_types_test=None,
    seed=0,
):
    train_row, train_cache = eval_split(
        model, model.rep_train, X_train, cond_train, device,
        split_name="train", cell_type=cell_types_train, compute_ksg=False, seed=seed
    )
    test_row, test_cache = eval_split(
        model, model.rep_test, X_test, cond_test, device,
        split_name="test", cell_type=cell_types_test, compute_ksg=False, seed=seed
    )

    base_df = pd.DataFrame([train_row, test_row]).set_index("split").T

    rows = {}
    rows.update(cond_pred_train_test(train_cache["X_2d"], cond_train, test_cache["X_2d"], cond_test, seed=seed, prefix="cond_from_X"))
    rows.update(cond_pred_train_test(train_cache["z"], cond_train, test_cache["z"], cond_test, seed=seed, prefix="cond_from_Z"))
    rows.update(cond_pred_train_test(train_cache["w_used"], cond_train, test_cache["w_used"], cond_test, seed=seed, prefix="cond_from_W"))
    rows.update(cond_pred_train_test(train_cache["xhat_joint_2d"], cond_train, test_cache["xhat_joint_2d"], cond_test, seed=seed, prefix="cond_from_xhat_joint"))
    rows.update(cond_pred_train_test(train_cache["xhat_zonly_2d"], cond_train, test_cache["xhat_zonly_2d"], cond_test, seed=seed, prefix="cond_from_xhat_zonly"))

    cond_df = pd.DataFrame(rows).T
    df_metrics = pd.concat([base_df, cond_df], axis=0)
    return df_metrics


def hpo_score_cond_from_Z_acc(model, X_train, cond_train, X_val, cond_val, device, seed=0):
    # train cache from rep_train, val cache from rep_test (since you trained val reps as "test")
    _, tr_cache = eval_split(model, model.rep_train, X_train, cond_train, device, split_name="train", seed=seed)
    _, va_cache = eval_split(model, model.rep_test,  X_val,   cond_val,   device, split_name="val",   seed=seed)

    # train logistic on Z(train), evaluate on Z(val)
    rows = cond_pred_train_test(tr_cache["z"], cond_train, va_cache["z"], cond_val, seed=seed, prefix="cond_from_Z")
    return rows["cond_from_Z_acc"]["test"]   # <- val acc


def _infer_cluster_keys(ds):
    # Heuristic defaults for FashionMNIST overlay
    z_key = None
    w_key = None
    if "fashion_label_train" in ds:
        z_key = "fashion_label"
    if "mnist_label_train" in ds:
        w_key = "mnist_label"
    return z_key, w_key


def final_test_metrics(model, ds_bundle, device, seed=0, extra_celltype_key=None,
                       clf_kind="knn", z_cluster_key=None, w_cluster_key=None):
    # ---- ensure trainval keys exist locally ----
    ds = dict(ds_bundle)

    if "X_trainval" not in ds:
        ds["X_trainval"] = concat_X(ds["X_train"], ds["X_val"])
    if "cond_trainval" not in ds:
        ds["cond_trainval"] = np.concatenate([ds["cond_train"], ds["cond_val"]], axis=0)

    X_tr = ds["X_trainval"]
    y_tr = np.asarray(ds["cond_trainval"]).ravel()

    X_te = ds["X_test"]
    y_te = np.asarray(ds["cond_test"]).ravel()

    cell_te = ds.get(extra_celltype_key, None) if extra_celltype_key else None

    train_row, train_cache = eval_split(
        model, model.rep_train, X_tr, y_tr, device, split_name="trainval", seed=seed
    )
    test_row, test_cache = eval_split(
        model, model.rep_test, X_te, y_te, device, split_name="test",
        cell_type=cell_te, seed=seed
    )

    clf_rows = {}
    clf_rows.update(cond_pred_train_test(train_cache["X_2d"], y_tr, test_cache["X_2d"], y_te, seed=seed, prefix="cond_from_X"))
    clf_rows.update(cond_pred_train_test(train_cache["z"], y_tr, test_cache["z"], y_te, seed=seed, prefix="cond_from_Z",
                                         kind=clf_kind, scale=True))
    clf_rows.update(cond_pred_train_test(train_cache["w_used"], y_tr, test_cache["w_used"], y_te, seed=seed, prefix="cond_from_W"))
    clf_rows.update(cond_pred_train_test(train_cache["xhat_joint_2d"], y_tr, test_cache["xhat_joint_2d"], y_te, seed=seed, prefix="cond_from_xhat_joint"))
    clf_rows.update(cond_pred_train_test(train_cache["xhat_zonly_2d"], y_tr, test_cache["xhat_zonly_2d"], y_te, seed=seed, prefix="cond_from_xhat_zonly"))

    out = {}

    # ---- eval_split(test) metrics ----
    for k, v in test_row.items():
        if k != "split":
            out[f"test/{k}"] = v

    # ---- predictors ----
    for metric_name, d in clf_rows.items():
        out[f"test/{metric_name}"] = d["test"]
        out[f"trainval/{metric_name}"] = d["train"]

    # ---- centered AUC ----
    auc_test = out.get("test/cond_from_Z_auc", np.nan)
    out["test/cond_from_Z_centered_auc"] = float(abs(float(auc_test) - 0.5)) if np.isfinite(auc_test) else np.nan

    # ---- ALWAYS return cluster MCC keys (defaults) ----
    out["test/z_cluster_mcc"] = 0.0
    out["test/w_cluster_mcc_active"] = 0.0

    # infer keys if not provided
    if z_cluster_key is None:
        z_cluster_key, w_cluster_key_infer = _infer_cluster_keys(ds)
        if w_cluster_key is None:
            w_cluster_key = w_cluster_key_infer

    # ---- Z cluster MCC ----
    if z_cluster_key is not None and f"{z_cluster_key}_trainval" in ds and f"{z_cluster_key}_test" in ds:
        zlab_trv = np.asarray(ds[f"{z_cluster_key}_trainval"]).ravel()
        zlab_te  = np.asarray(ds[f"{z_cluster_key}_test"]).ravel()
        if len(np.unique(zlab_trv)) > 1 and len(np.unique(zlab_te)) > 1:
            z_mcc = cluster_pred_train_test_mcc(
                train_cache["z"], zlab_trv,
                test_cache["z"],  zlab_te,
                seed=seed, kind=clf_kind
            )["mcc"]["test"]
            out["test/z_cluster_mcc"] = float(z_mcc)

    # ---- W cluster MCC (active only) ----
    if w_cluster_key is not None and f"{w_cluster_key}_trainval" in ds and f"{w_cluster_key}_test" in ds:
        wlab_trv = np.asarray(ds[f"{w_cluster_key}_trainval"]).ravel()
        wlab_te  = np.asarray(ds[f"{w_cluster_key}_test"]).ravel()

        y_trv = np.asarray(ds["cond_trainval"]).ravel()
        y_te2 = np.asarray(ds["cond_test"]).ravel()

        m_tr = (y_trv > 0.5) & (wlab_trv != -1)
        m_te = (y_te2 > 0.5) & (wlab_te  != -1)

        if (m_tr.sum() > 0) and (m_te.sum() > 0) and (len(np.unique(wlab_trv[m_tr])) > 1) and (len(np.unique(wlab_te[m_te])) > 1):
            w_mcc = cluster_pred_train_test_mcc(
                train_cache["w_used"][m_tr], wlab_trv[m_tr],
                test_cache["w_used"][m_te],  wlab_te[m_te],
                seed=seed, kind=clf_kind
            )["mcc"]["test"]
            out["test/w_cluster_mcc_active"] = float(w_mcc)

    return out
