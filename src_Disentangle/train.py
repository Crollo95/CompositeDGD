import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import DGDModel
from .data import TabularDataset
from .latent import SplitRepresentationLayer


def train_dgd(
    X_train,
    X_test,
    cond_train,
    cond_test,
    z_prior_on_y0_only:bool=True,
    n_z=8,
    n_w=8,
    n_components_z=8,
    n_components_w=8,
    beta_w=1.0,
    beta_w_anneal: bool = False,
    hidden_dims=(64, 64),
    output_module: str = "gaussian",
    output_module_options=None,
    weights_prior_options_z: dict | None = None,
    weights_prior_options_w: dict | None = None,
    batch_size=64,
    lr_prior=1e-3,
    lr_decoder=1e-3,
    lr_rep_train=1e-2,
    lr_rep_test=1e-2,
    n_epochs=200,
    dropout_p=0.0,
    film: bool = False,
    log_transform=False,
    eps=1e-6,
    standardize=True,
    device="cpu",
):
  
    device = torch.device(device)

    is_image = (X_train.ndim == 4)
    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]
    
    if is_image:
        # CNN path
        img_shape = tuple(X_train.shape[1:])   # (C,H,W)
        n_features = int(torch.tensor(img_shape).prod().item())  # only to satisfy constructor
    else:
        # tabular path
        n_features = X_train.shape[1]

    # --------- preprocess TRAIN ---------
    
    idx_train = torch.arange(n_train, dtype=torch.long)
    cond_train = torch.as_tensor(cond_train, dtype=torch.float32)

    
    # --------- global scaling factors (NB softmax only) ---------
    _use_global_sf = (
        not is_image
        and output_module == "nb"
        and isinstance(output_module_options, dict)
        and output_module_options.get("mean_mode") == "softmax"
    )
    if _use_global_sf:
        _sf_raw_train = torch.as_tensor(X_train, dtype=torch.float32).sum(dim=1)  # (n_train,)
        _sf_raw_test  = torch.as_tensor(X_test,  dtype=torch.float32).sum(dim=1)  # (n_test,)
        _global_median = float(_sf_raw_train.median()) + 1e-8
        sf_train = _sf_raw_train / _global_median   # (n_train,)
        sf_test  = _sf_raw_test  / _global_median   # (n_test,) — same median as train

    if is_image:
        Xtr = torch.as_tensor(X_train, dtype=torch.float32)
        ds_train = TensorDataset(Xtr, idx_train, cond_train)
        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

        idx_test = torch.arange(n_test, dtype=torch.long)
        cond_test = torch.as_tensor(cond_test, dtype=torch.float32)
        Xte = torch.as_tensor(X_test, dtype=torch.float32)
        ds_test = TensorDataset(Xte, idx_test, cond_test)
        loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

        base_train = None
        base_test = None

    else:
        base_train = TabularDataset(
            X_train,
            log_transform=log_transform,
            eps=eps,
            standardize=standardize,
        )

        if _use_global_sf:
            ds_train = TensorDataset(base_train.X, idx_train, cond_train, sf_train)
        else:
            ds_train = TensorDataset(base_train.X, idx_train, cond_train)
        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

        # --------- preprocess TEST using TRAIN stats ---------
        base_test = TabularDataset(
            X_test,
            log_transform=base_train.log_transform,
            eps=base_train.eps,
            standardize=base_train.standardize,
            mean=base_train.feature_mean,
            std=base_train.feature_std,
        )
        idx_test = torch.arange(n_test, dtype=torch.long)
        cond_test = torch.as_tensor(cond_test, dtype=torch.float32)

        if _use_global_sf:
            ds_test = TensorDataset(base_test.X, idx_test, cond_test, sf_test)
        else:
            ds_test = TensorDataset(base_test.X, idx_test, cond_test)
        loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    
    # --------- model ---------
    model = DGDModel(
        n_features=n_features,
        n_z=n_z,
        n_w=n_w,
        n_components_z=n_components_z,
        n_components_w=n_components_w,
        beta_w=beta_w,
        hidden_dims=hidden_dims,
        output_module=output_module,
        output_module_options=output_module_options,
        weights_prior_options_z=weights_prior_options_z,
        weights_prior_options_w=weights_prior_options_w,
        dropout_p=dropout_p,
        film=film,
        device=device,

        decoder_type=("cnn" if is_image else "mlp"),
        img_shape=(img_shape if is_image else (1, 28, 28)),
        cnn_likelihood="bernoulli",
    )
    model.train()

    # --------- representation layers ---------
    rep_train = SplitRepresentationLayer(n_samples=n_train, n_z=n_z, n_w=n_w,).to(device)
    rep_test  = SplitRepresentationLayer(n_samples=n_test,  n_z=n_z, n_w=n_w,).to(device) 

    # --- make stored w literally zero for cond=0 rows ---
    with torch.no_grad():
        m0_tr = (cond_train <= 0.5).to(device)
        rep_train.w[m0_tr] = 0.0
        m0_te = (cond_test <= 0.5).to(device)
        rep_test.w[m0_te] = 0.0
    
    # --------- optimizers ---------
    opt_prior = torch.optim.Adam(
        list(model.prior_z.parameters()) + list(model.prior_w.parameters()), lr=lr_prior
    )
    opt_decoder   = torch.optim.Adam(model.decoder.parameters(), lr=lr_decoder, weight_decay=1e-4)
    opt_rep_train = torch.optim.Adam(rep_train.parameters(),     lr=lr_rep_train)
    opt_rep_test  = torch.optim.Adam(rep_test.parameters(),      lr=lr_rep_test)

    # --------- history containers ---------
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_prior": [],
        "train_prior_latent": [],
        "train_prior_params": [],
    
        "train_prior_z_latent": [],
        "train_prior_z_params": [],
        "train_prior_w_latent": [],
        "train_prior_w_params": [],
    
        "test_loss": [],
        "test_recon": [],
        "test_prior": [],
        "test_prior_latent": [],
        "test_prior_params": [],
    
        "test_prior_z_latent": [],
        "test_prior_z_params": [],
        "test_prior_w_latent": [],
        "test_prior_w_params": [],
    
        "train_recon_components": [],
        "test_recon_components": [],
    }

    # --------- beta_w annealing setup ---------
    beta_w_target = model.beta_w
    if beta_w_anneal:
        warmup_epochs = max(1, int(0.2 * n_epochs))

    # --------- training loop ---------
    for epoch in range(1, n_epochs + 1):
        if beta_w_anneal:
            model.beta_w = beta_w_target * min(1.0, epoch / warmup_epochs)
        # ===== TRAIN PHASE =====
        model.train()
        rep_train.train()

        total_loss = 0.0
        total_recon = 0.0
        total_prior = 0.0
        total_prior_latent = 0.0
        total_prior_params = 0.0

        total_prior_z_latent  = 0.0
        total_prior_z_params  = 0.0
        total_prior_w_latent  = 0.0
        total_prior_w_params  = 0.0

        # accumulate per-module recon losses
        comp_sums: dict[str, float] = {}
        opt_rep_train.zero_grad()
        
        for batch in loader_train:
            if _use_global_sf:
                batch_y, batch_idx, batch_cond, batch_sf = batch
                batch_sf = batch_sf.to(device)
            else:
                batch_y, batch_idx, batch_cond = batch
                batch_sf = None
            batch_y = batch_y.to(device)
            batch_idx = batch_idx.to(device)
            batch_cond = batch_cond.to(device)

            opt_prior.zero_grad()
            opt_decoder.zero_grad()
            #opt_rep_train.zero_grad()

            (
                loss,
                recon_loss,
                prior_loss,
                recon_components,
                prior_z_nll,
                prior_z_params,
                prior_w_nll,
                prior_w_params,
            ) = model.batch_loss(
                batch_y,
                rep_layer=rep_train,
                batch_indices=batch_idx,
                cond_mask=batch_cond,
                z_prior_on_y0_only=z_prior_on_y0_only,
                scaling_factor=batch_sf,
            )
            
            loss.backward()

            opt_prior.step()
            opt_decoder.step()
            #opt_rep_train.step()

            bs = batch_y.shape[0]
            total_loss         += loss.item()        * bs
            total_recon        += recon_loss.item()  * bs
            total_prior        += prior_loss.item()  * bs

            total_prior_z_latent += prior_z_nll.item() * bs
            total_prior_z_params += prior_z_params.item() * bs
            total_prior_w_latent += prior_w_nll.item() * bs
            total_prior_w_params += prior_w_params.item() * bs

            # keep old history keys: "prior_z" and "prior_params" as sums over both priors
            total_prior_latent += (prior_z_nll.item() + prior_w_nll.item()) * bs
            total_prior_params += (prior_z_params.item() + prior_w_params.item()) * bs


            # per-module losses: each is a scalar tensor
            for name, nll_tensor in recon_components.items():
                comp_sums[name] = comp_sums.get(name, 0.0) + nll_tensor.item() * bs

        opt_rep_train.step() 

        
        avg_loss         = total_loss         / n_train
        avg_recon        = total_recon        / n_train
        avg_prior        = total_prior        / n_train
        avg_prior_latent = total_prior_latent / n_train
        avg_prior_params = total_prior_params / n_train

        avg_prior_z_latent = total_prior_z_latent / n_train
        avg_prior_z_params = total_prior_z_params / n_train
        avg_prior_w_latent = total_prior_w_latent / n_train
        avg_prior_w_params = total_prior_w_params / n_train


        # compute epoch-wise per-module averages
        train_comp_avgs = {name: s / n_train for name, s in comp_sums.items()}

        history["train_loss"].append(avg_loss)
        history["train_recon"].append(avg_recon)
        history["train_prior"].append(avg_prior)
        history["train_prior_latent"].append(avg_prior_latent)
        history["train_prior_params"].append(avg_prior_params)
        history["train_recon_components"].append(train_comp_avgs)

        history["train_prior_z_latent"].append(avg_prior_z_latent)
        history["train_prior_z_params"].append(avg_prior_z_params)
        history["train_prior_w_latent"].append(avg_prior_w_latent)
        history["train_prior_w_params"].append(avg_prior_w_params)


        # ===== TEST REP PHASE =====
        model.eval()
        rep_test.train()  # reps still learn

        # Freeze model params
        for p in model.prior_z.parameters():
            p.requires_grad_(False)
        for p in model.prior_w.parameters():
            p.requires_grad_(False)
        for p in model.decoder.parameters():
            p.requires_grad_(False)

        total_test_loss = 0.0
        total_test_recon = 0.0
        total_test_prior = 0.0
        total_test_prior_latent = 0.0
        total_test_prior_params = 0.0
        
        total_test_prior_z_latent = 0.0
        total_test_prior_z_params = 0.0
        total_test_prior_w_latent = 0.0
        total_test_prior_w_params = 0.0

        comp_test_sums: dict[str, float] = {}

        opt_rep_test.zero_grad()
        
        for batch_test in loader_test:
            if _use_global_sf:
                batch_y_test, batch_idx_test, batch_cond_test, batch_sf_test = batch_test
                batch_sf_test = batch_sf_test.to(device)
            else:
                batch_y_test, batch_idx_test, batch_cond_test = batch_test
                batch_sf_test = None
            batch_y_test = batch_y_test.to(device)
            batch_idx_test = batch_idx_test.to(device)
            batch_cond_test = batch_cond_test.to(device)

            #opt_rep_test.zero_grad()

            (
                loss_test,
                recon_test,
                prior_test,
                recon_comp_test,
                prior_z_nll_t,
                prior_z_params_t,
                prior_w_nll_t,
                prior_w_params_t,
            ) = model.batch_loss(
                batch_y_test,
                rep_layer=rep_test,
                batch_indices=batch_idx_test,
                cond_mask=batch_cond_test,
                z_prior_on_y0_only=z_prior_on_y0_only,
                scaling_factor=batch_sf_test,
            )

            loss_test.backward()
            #opt_rep_test.step()

            bs = batch_y_test.shape[0]
            total_test_loss         += loss_test.item()        * bs
            total_test_recon        += recon_test.item()       * bs
            total_test_prior        += prior_test.item()       * bs

            total_test_prior_z_latent += prior_z_nll_t.item() * bs
            total_test_prior_z_params += prior_z_params_t.item() * bs
            total_test_prior_w_latent += prior_w_nll_t.item() * bs
            total_test_prior_w_params += prior_w_params_t.item() * bs

            total_test_prior_latent += (prior_z_nll_t.item() + prior_w_nll_t.item()) * bs
            total_test_prior_params += (prior_z_params_t.item() + prior_w_params_t.item()) * bs

            for name, nll_tensor in recon_comp_test.items():
                comp_test_sums[name] = comp_test_sums.get(name, 0.0) + nll_tensor.item() * bs
                

        opt_rep_test.step()
        
        avg_test_loss         = total_test_loss         / n_test
        avg_test_recon        = total_test_recon        / n_test
        avg_test_prior        = total_test_prior        / n_test
        avg_test_prior_latent = total_test_prior_latent / n_test
        avg_test_prior_params = total_test_prior_params / n_test

        avg_test_prior_z_latent = total_test_prior_z_latent / n_test
        avg_test_prior_z_params = total_test_prior_z_params / n_test
        avg_test_prior_w_latent = total_test_prior_w_latent / n_test
        avg_test_prior_w_params = total_test_prior_w_params / n_test


        test_comp_avgs = {name: s / n_test for name, s in comp_test_sums.items()}

        history["test_loss"].append(avg_test_loss)
        history["test_recon"].append(avg_test_recon)
        history["test_prior"].append(avg_test_prior)
        history["test_prior_latent"].append(avg_test_prior_latent)
        history["test_prior_params"].append(avg_test_prior_params)
        history["test_recon_components"].append(test_comp_avgs)

        history["test_prior_z_latent"].append(avg_test_prior_z_latent)
        history["test_prior_z_params"].append(avg_test_prior_z_params)
        history["test_prior_w_latent"].append(avg_test_prior_w_latent)
        history["test_prior_w_params"].append(avg_test_prior_w_params)


        # Unfreeze model params
        for p in model.prior_z.parameters():
            p.requires_grad_(True)
        for p in model.prior_w.parameters():
            p.requires_grad_(True)
        for p in model.decoder.parameters():
            p.requires_grad_(True)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={avg_loss:.4f} (recon={avg_recon:.4f}, prior={avg_prior:.4f}) | "
            f"z: latent={avg_prior_z_latent:.3f} params={avg_prior_z_params:.3f} | "
            f"w: latent={avg_prior_w_latent:.3f} params={avg_prior_w_params:.3f} | "
            f"test loss={avg_test_loss:.4f} (recon={avg_test_recon:.4f}, prior={avg_test_prior:.4f}) | "
            f"z: latent={avg_test_prior_z_latent:.3f} params={avg_test_prior_z_params:.3f} | "
            f"w: latent={avg_test_prior_w_latent:.3f} params={avg_test_prior_w_params:.3f}"
        )



    # Restore full beta_w after annealing
    if beta_w_anneal:
        model.beta_w = beta_w_target

    # Attach reps & preprocessing to model
    model.rep_train = rep_train
    model.rep_test = rep_test
    if not is_image:
        model.preproc_mean = base_train.feature_mean
        model.preproc_std  = base_train.feature_std
        model.log_transform = base_train.log_transform
        model.eps = base_train.eps
        

    return model, history

