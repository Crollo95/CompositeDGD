import numpy as np
from src.metrics_eval import get_latents_split, infer_cluster_keys
from src.metrics_predictors import DEFAULT_CLF_KIND, cond_pred_train_test, cluster_pred_train_test_mcc

def score_on_validation(model, ds_bundle, device, seed=0, penalty=1.0, centered_auc=True, clf_kind=DEFAULT_CLF_KIND):
    y_tr  = np.asarray(ds_bundle["cond_train"]).ravel()
    y_val = np.asarray(ds_bundle["cond_val"]).ravel()

    # must have both classes
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
        return float(penalty)

    try:
        z_tr, _  = get_latents_split(model.rep_train, y_tr,  device=device)
        z_val, _ = get_latents_split(model.rep_test,  y_val, device=device)

        if (not np.isfinite(z_tr).all()) or (not np.isfinite(z_val).all()):
            return float(penalty)

        rows = cond_pred_train_test(
            z_tr, y_tr,
            z_val, y_val,
            seed=seed,
            prefix="cond_from_Z",
            kind=clf_kind,
            scale=True,
        )

        auc = float(rows["cond_from_Z_auc"]["test"])
        if not np.isfinite(auc):
            return float(penalty)

        return float(abs(auc - 0.5)) if centered_auc else float(auc)

    except Exception:
        return float(penalty)
        

def z_cluster_mcc_on_val(model, ds_bundle, device, seed=0, z_cluster_key=None, kind=DEFAULT_CLF_KIND):
    if z_cluster_key is None:
        z_cluster_key, _ = infer_cluster_keys(ds_bundle)
    if z_cluster_key is None:
        return 0.0

    ktr = f"{z_cluster_key}_train"
    kva = f"{z_cluster_key}_val"
    if (ktr not in ds_bundle) or (kva not in ds_bundle):
        return 0.0

    y_tr = np.asarray(ds_bundle[ktr]).ravel()
    y_va = np.asarray(ds_bundle[kva]).ravel()

    # need >1 class
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        return 0.0

    cond_tr = np.asarray(ds_bundle["cond_train"]).ravel()
    cond_va = np.asarray(ds_bundle["cond_val"]).ravel()

    z_tr, _ = get_latents_split(model.rep_train, cond_tr, device=device)
    z_va, _ = get_latents_split(model.rep_test,  cond_va, device=device)

    if (not np.isfinite(z_tr).all()) or (not np.isfinite(z_va).all()):
        return 0.0

    return float(cluster_pred_train_test_mcc(
        z_tr, y_tr, z_va, y_va, seed=seed, kind=kind
    )["mcc"]["test"])


def w_cluster_mcc_on_val(model, ds_bundle, device, seed=0, w_cluster_key=None, kind=DEFAULT_CLF_KIND):
    if w_cluster_key is None:
        _, w_cluster_key = infer_cluster_keys(ds_bundle)
    if w_cluster_key is None:
        return 0.0

    ktr = f"{w_cluster_key}_train"
    kva = f"{w_cluster_key}_val"
    if (ktr not in ds_bundle) or (kva not in ds_bundle):
        return 0.0

    wlab_tr = np.asarray(ds_bundle[ktr]).ravel()
    wlab_va = np.asarray(ds_bundle[kva]).ravel()

    cond_tr = np.asarray(ds_bundle["cond_train"]).ravel()
    cond_va = np.asarray(ds_bundle["cond_val"]).ravel()

    _, w_tr = get_latents_split(model.rep_train, cond_tr, device=device)
    _, w_va = get_latents_split(model.rep_test,  cond_va, device=device)

    # cond=1 only + ignore label -1 if present
    m_tr = (cond_tr > 0.5) & (wlab_tr != -1)
    m_va = (cond_va > 0.5) & (wlab_va != -1)

    if m_tr.sum() == 0 or m_va.sum() == 0:
        return 0.0

    if len(np.unique(wlab_tr[m_tr])) < 2 or len(np.unique(wlab_va[m_va])) < 2:
        return 0.0

    Wtr = w_tr[m_tr]; Ytr = wlab_tr[m_tr]
    Wva = w_va[m_va]; Yva = wlab_va[m_va]

    if (not np.isfinite(Wtr).all()) or (not np.isfinite(Wva).all()):
        return 0.0

    return float(cluster_pred_train_test_mcc(
        Wtr, Ytr, Wva, Yva, seed=seed, kind=kind
    )["mcc"]["test"])