import copy
import numpy as np
import torch
import optuna

from src.train import train_dgd
from src.hpo_objectives import score_on_validation, z_cluster_mcc_on_val, w_cluster_mcc_on_val
from src.metrics_eval import evaluate_model, get_latents_split, concat_X


def _pick_trial(study):
    """
    Choose ONE trial for retraining from non-dominated trials:
      1) minimize val_score
      2) maximize z_mcc
      3) maximize w_mcc
    """
    pareto = [t for t in study.best_trials
              if t.values is not None
              and all(v is not None for v in t.values)]
    if not pareto:
        pareto = [t for t in study.trials
                  if t.state == optuna.trial.TrialState.COMPLETE
                  and t.values is not None and all(v is not None for v in t.values)]
    return min(pareto, key=lambda t: (t.values[0], -t.values[1], -t.values[2]))


def _pick_trial_balanced(study, use_pareto=True, norm_on="pareto", distance="l2", weights=(1.0, 1.0, 1.0)):
    """
    Balanced selection among multi-objective trials.

    Objectives (your setup):
      v0 = val_score (MINIMIZE)
      v1 = z_mcc     (MAXIMIZE)
      v2 = w_mcc     (MAXIMIZE)

    Strategy:
      1) pick candidate set (Pareto front by default)
      2) normalize each objective to [0,1] using min/max across 'pareto' or 'all'
      3) convert to "losses" to minimize, where 0 is ideal:
           loss0 = norm(v0)                 (smaller is better)
           loss1 = 1 - norm(v1)             (bigger is better => smaller loss)
           loss2 = 1 - norm(v2)
      4) pick trial minimizing weighted distance to (0,0,0)
    """
    # --- candidates ---
    if use_pareto and getattr(study, "best_trials", None):
        candidates = [t for t in study.best_trials if t.values is not None and all(v is not None for v in t.values)]
    else:
        candidates = []

    if not candidates:
        candidates = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.values is not None
            and all(v is not None for v in t.values)
        ]
    if not candidates:
        return study.best_trials[0]  # fallback

    # --- normalization set ---
    if norm_on == "all":
        norm_set = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.values is not None
            and all(v is not None for v in t.values)
        ]
        if not norm_set:
            norm_set = candidates
    else:
        norm_set = candidates

    V = np.array([t.values for t in norm_set], dtype=float)  # shape (N,3)
    mins = V.min(axis=0)
    maxs = V.max(axis=0)
    rng  = np.maximum(maxs - mins, 1e-12)  # avoid div-by-zero

    w0, w1, w2 = map(float, weights)

    def score(trial):
        v0, v1, v2 = map(float, trial.values)

        # normalize to [0,1]
        n0 = (v0 - mins[0]) / rng[0]  # minimize
        n1 = (v1 - mins[1]) / rng[1]  # maximize
        n2 = (v2 - mins[2]) / rng[2]  # maximize

        # convert to losses to minimize (0 = ideal)
        l0 = n0
        l1 = 1.0 - n1
        l2 = 1.0 - n2

        if distance == "l1":
            return w0*abs(l0) + w1*abs(l1) + w2*abs(l2)
        else:  # l2
            return np.sqrt((w0*l0)**2 + (w1*l1)**2 + (w2*l2)**2)

    return min(candidates, key=score)



def build_latents_df(final_model, ds_tv, device, z_cluster_key=None, w_cluster_key=None):
    import pandas as pd

    out = {}
    meta = ds_tv.get("meta", {}) if isinstance(ds_tv, dict) else {}

    for split_name in ["trainval", "test"]:
        y_key = f"cond_{split_name}"
        if y_key not in ds_tv:
            continue

        cond = np.asarray(ds_tv[y_key]).ravel()
        rep_layer = final_model.rep_train if split_name == "trainval" else final_model.rep_test
        z, w = get_latents_split(rep_layer, cond, device=device)

        # always available, always consistent
        df = pd.DataFrame({
            "split_index": np.arange(len(cond), dtype=np.int64),
            "cond": cond.astype(np.float32),
        })

        # ---- PREFER string/barcode indices_{split} if present ----
        idx_key = f"indices_{split_name}"
        if idx_key in ds_tv:
            orig_idx = np.asarray(ds_tv[idx_key])
            if len(orig_idx) == len(cond):
                df["orig_index"] = orig_idx  # keep dtype as-is (strings/objects)
        else:
            # fallback to integer indices from meta (if available AND length matches)
            orig_idx = None
            if split_name == "trainval" and ("idx_train" in meta) and ("idx_val" in meta):
                orig_idx = np.concatenate([np.asarray(meta["idx_train"]), np.asarray(meta["idx_val"])])
            elif split_name == "test" and ("idx_test" in meta):
                orig_idx = np.asarray(meta["idx_test"])

            if orig_idx is not None and len(orig_idx) == len(cond):
                df["orig_index"] = orig_idx.astype(np.int64)

        # z / w columns
        for j in range(z.shape[1]):
            df[f"z_{j}"] = z[:, j]
        for j in range(w.shape[1]):
            df[f"w_{j}"] = w[:, j]

        # optional clusters
        if z_cluster_key is not None:
            k = f"{z_cluster_key}_{split_name}"
            df["z_cluster"] = np.asarray(ds_tv[k]).ravel() if k in ds_tv else -1

        if w_cluster_key is not None:
            k = f"{w_cluster_key}_{split_name}"
            if k in ds_tv:
                wlab = np.asarray(ds_tv[k]).ravel().copy()
                wlab[cond <= 0.5] = -1
                df["w_cluster"] = wlab
            else:
                df["w_cluster"] = -1

        out[split_name] = df

    return out



def run_one_split_experiment_optuna(
    ds_bundle,
    device,
    base_kw,
    split_seed=0,
    n_trials=20,
    timeout=None,
    verbose=True,
    clf_kind=None,       # non-linear choice
    z_cluster_key=None,      # e.g. "fashion_label" or "cell_types"
    w_cluster_key=None,      # e.g. "mnist_label" or "groups"
    hp_space=None,       # callable (trial) -> dict; None = default space
):
    def _default_hp_space(trial):
        lr_rep = trial.suggest_float("lr_rep", 5e-3, 3e-2, log=True)
        return {
            "n_z": trial.suggest_int("n_z", 8, 16, step=4),
            "n_w": trial.suggest_int("n_w", 8, 16, step=4),
            "lr_decoder": trial.suggest_float("lr_decoder", 1e-4, 1e-2, log=True),
            "lr_prior":   trial.suggest_float("lr_prior",   1e-4, 5e-3, log=True),
            "lr_rep_train": lr_rep,
            "lr_rep_test": lr_rep,
            "beta_w": trial.suggest_categorical("beta_w", [0.5, 1.0, 1.5]),
            "n_epochs": trial.suggest_int("n_epochs", 200, 600, step=20),
            "film": trial.suggest_categorical("film", [False, True]),
        }

    _suggest = hp_space if hp_space is not None else _default_hp_space

    def objective(trial: optuna.Trial):
        hp = _suggest(trial)

        kw = copy.deepcopy(base_kw)
        kw.update(hp)

        seed = int(split_seed * 10_000 + trial.number)
        torch.manual_seed(seed)
        np.random.seed(seed)

        try:
            model, history = train_dgd(
                X_train=ds_bundle["X_train"],
                X_test=ds_bundle["X_val"],
                cond_train=ds_bundle["cond_train"],
                cond_test=ds_bundle["cond_val"],
                device=device,
                **kw,
            )

            # divergence quick check
            if (len(history["train_loss"]) > 0) and (not np.isfinite(history["train_loss"][-1])):
                return (1.0, -1.0, -1.0)

            # 1) disentanglement objective (minimize)
            val_score = score_on_validation(
                model, ds_bundle, device=device, seed=seed,
                penalty=1.0, centered_auc=True, clf_kind=clf_kind,
            )
            val_score = round(val_score,3)
            
            # 2) Z-cluster objective (maximize MCC)
            z_mcc = z_cluster_mcc_on_val(
                model, ds_bundle, device=device, seed=seed,
                z_cluster_key=z_cluster_key, kind=clf_kind
            )
            z_mcc = round(z_mcc, 3)

            # 3) W-cluster objective (maximize MCC; cond=1 only)
            w_mcc = w_cluster_mcc_on_val(
                model, ds_bundle, device=device, seed=seed,
                w_cluster_key=w_cluster_key, kind=clf_kind
            )
            w_mcc = round(w_mcc, 3)

            trial.set_user_attr("val_score", float(val_score))
            trial.set_user_attr("z_mcc", float(z_mcc))
            trial.set_user_attr("w_mcc", float(w_mcc))

            if verbose:
                print(f"[split {split_seed}][trial {trial.number}] val={val_score:.4f} z_mcc={z_mcc:.3f} w_mcc={w_mcc:.3f} hp={hp}")

            return (float(val_score), float(z_mcc), float(w_mcc))

        except Exception as e:
            trial.set_user_attr("failed", repr(e))
            if verbose:
                print(f"[split {split_seed}][trial {trial.number}] FAILED: {repr(e)}")
            return (1.0, -1.0, -1.0)

    sampler = optuna.samplers.NSGAIISampler(seed=int(split_seed))
    study = optuna.create_study(directions=["minimize", "maximize", "maximize"], sampler=sampler)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose,
        catch=(Exception,),
    )

    #chosen = _pick_trial(study)
    chosen = _pick_trial_balanced(study, weights = (1.5,1.0,1.0))
    best_hp = chosen.params
    val_score, z_mcc, w_mcc = chosen.values

    if verbose:
        print(f"[split {split_seed}] CHOSEN HP: {best_hp} | val={val_score:.4f} z_mcc={z_mcc:.3f} w_mcc={w_mcc:.3f}")

    # retrain on train+val
    X_trval = concat_X(ds_bundle["X_train"], ds_bundle["X_val"])
    y_trval = np.concatenate([ds_bundle["cond_train"], ds_bundle["cond_val"]], axis=0)

    kw_final = copy.deepcopy(base_kw)
    kw_final.update(best_hp)

    if "lr_rep" in kw_final:                                                                                                                         
        lr_rep = kw_final.pop("lr_rep")                                                                                                              
        kw_final.setdefault("lr_rep_train", lr_rep)                                                                                                  
        kw_final.setdefault("lr_rep_test", lr_rep)   

    torch.manual_seed(int(split_seed + 10_000))
    np.random.seed(int(split_seed + 10_000))

    final_model, final_history = train_dgd(
        X_train=X_trval,
        X_test=ds_bundle["X_test"],
        cond_train=y_trval,
        cond_test=ds_bundle["cond_test"],
        device=device,
        **kw_final,
    )

    ds_tv = ensure_trainval_keys(ds_bundle)

    test_metrics = evaluate_model(
        final_model, ds_tv, device=device, seed=int(split_seed + 99_999),
        clf_kind=clf_kind,
        z_cluster_key=z_cluster_key,
        w_cluster_key=w_cluster_key,
    )

    return {
        "best_hp": best_hp,
        "val_score": float(val_score),
        "val_z_mcc": float(z_mcc),
        "val_w_mcc": float(w_mcc),
        "study": study,
        "final_model": final_model,
        "final_history": final_history,
        "test_metrics": test_metrics,
    }


def run_experiments_optuna(
    dataset_loader,
    dataset_cfg,
    n_splits=1,
    base_kw=None,
    z_cluster_key=None,
    w_cluster_key=None,
    clf_kind="rf",
    device="cpu",
    n_trials=20,
    timeout=None,
):
    device = torch.device(device)
    results = []

    for split_seed in range(n_splits):
        ds = dataset_loader(dataset_cfg, split_seed=split_seed)

        out = run_one_split_experiment_optuna(
            ds_bundle=ds,
            device=device,
            base_kw=base_kw,
            split_seed=split_seed,
            n_trials=n_trials,
            timeout=timeout,
            verbose=True,
            clf_kind=clf_kind,
            z_cluster_key=z_cluster_key,
            w_cluster_key=w_cluster_key,
        )

        ds_tv = ensure_trainval_keys(ds)

        latents_df = build_latents_df(
            out["final_model"], ds_tv, device=device,
            z_cluster_key=z_cluster_key,
            w_cluster_key=w_cluster_key,
        ),

        results.append({
            "split_seed": int(split_seed),
            "sizes": {
                "train": int(len(ds_tv["cond_train"])),
                "val": int(len(ds_tv["cond_val"])),
                "trainval": int(len(ds_tv["cond_trainval"])),
                "test": int(len(ds_tv["cond_test"])),
            },
            "best_hp": dict(out["best_hp"]),
            "val_score": float(out["val_score"]),
            "val_z_mcc": float(out.get("val_z_mcc", 0.0)),
            "val_w_mcc": float(out.get("val_w_mcc", 0.0)),
            "test_metrics": out["test_metrics"],
            "latents_df" : latents_df,
            "study": out.get("study", None),
            "final_history": out.get("final_history", None),
            "final_model": out.get("final_model", None),
        })

    return results
    


def ensure_trainval_keys(ds_bundle):
    """
    Make ds_bundle contain:
      - X_trainval, cond_trainval
    Also, if it has any *_train and *_val arrays (e.g. cell_types_train),
    it will create *_trainval too.
    """
    ds = dict(ds_bundle)  # shallow copy

    # core
    if "X_trainval" not in ds:
        ds["X_trainval"] = concat_X(ds["X_train"], ds["X_val"])
    if "cond_trainval" not in ds:
        ds["cond_trainval"] = np.concatenate([ds["cond_train"], ds["cond_val"]], axis=0)

    # extras: auto-merge *_train + *_val -> *_trainval
    for k in list(ds.keys()):
        if k.endswith("_train"):
            base = k[:-6]
            k_val = base + "_val"
            k_tv  = base + "_trainval"
            if (k_val in ds) and (k_tv not in ds):
                ds[k_tv] = np.concatenate([np.asarray(ds[k]), np.asarray(ds[k_val])], axis=0)

    return ds