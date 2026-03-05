import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import DGDModel
from .data import TabularDataset
from .latent import RepresentationLayer


def train_dgd(
    X_train,
    X_test,
    latent_dim=16,
    n_components=8,
    hidden_dims=(64, 64),
    output_module: str = "gaussian",
    output_module_options=None,
    weights_prior_options: dict | None = None,
    batch_size=64,
    lr_prior=1e-3,
    lr_decoder=1e-3,
    lr_rep_train=1e-2,
    lr_rep_test=1e-2,
    n_epochs=200,
    dropout_p=0.0,
    log_transform=False,
    eps=1e-6,
    standardize=True,
    device="cpu",
):
    device = torch.device(device)

    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]

    # --------- preprocess TRAIN ---------
    base_train = TabularDataset(
        X_train,
        log_transform=log_transform,
        eps=eps,
        standardize=standardize,
    )
    idx_train = torch.arange(n_train, dtype=torch.long)
    ds_train = TensorDataset(base_train.X, idx_train)
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
    ds_test = TensorDataset(base_test.X, idx_test)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    # --------- model ---------
    model = DGDModel(
        n_features=n_features,
        latent_dim=latent_dim,
        n_components=n_components,
        hidden_dims=hidden_dims,
        output_module=output_module,
        output_module_options=output_module_options,
        weights_prior_options=weights_prior_options,
        dropout_p=dropout_p,
        device=device,
    )
    model.train()

    # --------- representation layers ---------
    rep_train = RepresentationLayer(n_samples=n_train, latent_dim=latent_dim).to(device)
    rep_test  = RepresentationLayer(n_samples=n_test,  latent_dim=latent_dim).to(device)

    # --------- optimizers ---------
    opt_prior     = torch.optim.Adam(model.prior.parameters(),   lr=lr_prior)
    opt_decoder   = torch.optim.Adam(model.decoder.parameters(), lr=lr_decoder)
    opt_rep_train = torch.optim.Adam(rep_train.parameters(),     lr=lr_rep_train)
    opt_rep_test  = torch.optim.Adam(rep_test.parameters(),      lr=lr_rep_test)

    # --------- history containers ---------
    history = {
        "train_loss": [],
        "train_recon": [],
        "train_prior": [],         # total prior (z + params)
        "train_prior_z": [],       # just -E[log p(z | GMM)]
        "train_prior_params": [],  # just -log p(parameters)/batch
        "test_loss": [],
        "test_recon": [],
        "test_prior": [],
        "test_prior_z": [],
        "test_prior_params": [],
        "train_recon_components": [],
        "test_recon_components": [],
    }

    # --------- training loop ---------
    for epoch in range(1, n_epochs + 1):
        # ===== TRAIN PHASE =====
        model.train()
        rep_train.train()

        total_loss = 0.0
        total_recon = 0.0
        total_prior = 0.0
        total_prior_z = 0.0
        total_prior_params = 0.0

        # accumulate per-module recon losses
        comp_sums: dict[str, float] = {}

        for batch_y, batch_idx in loader_train:
            batch_y = batch_y.to(device)
            batch_idx = batch_idx.to(device)

            opt_prior.zero_grad()
            opt_decoder.zero_grad()
            opt_rep_train.zero_grad()

            (
                loss,
                recon_loss,
                prior_loss,
                recon_components,
                prior_z,
                prior_params,
            ) = model.batch_loss(
                batch_y, rep_layer=rep_train, batch_indices=batch_idx
            )

            loss.backward()

            opt_prior.step()
            opt_decoder.step()
            opt_rep_train.step()

            bs = batch_y.shape[0]
            total_loss         += loss.item()        * bs
            total_recon        += recon_loss.item()  * bs
            total_prior        += prior_loss.item()  * bs
            total_prior_z      += prior_z.item()     * bs
            total_prior_params += prior_params.item()* bs

            # per-module losses: each is a scalar tensor
            for name, nll_tensor in recon_components.items():
                comp_sums[name] = comp_sums.get(name, 0.0) + nll_tensor.item() * bs

        #opt_rep_train.step() 

        
        avg_loss         = total_loss         / n_train
        avg_recon        = total_recon       / n_train
        avg_prior        = total_prior       / n_train
        avg_prior_z      = total_prior_z     / n_train
        avg_prior_params = total_prior_params/ n_train

        # compute epoch-wise per-module averages
        train_comp_avgs = {name: s / n_train for name, s in comp_sums.items()}

        history["train_loss"].append(avg_loss)
        history["train_recon"].append(avg_recon)
        history["train_prior"].append(avg_prior)
        history["train_prior_z"].append(avg_prior_z)
        history["train_prior_params"].append(avg_prior_params)
        history["train_recon_components"].append(train_comp_avgs)

        # ===== TEST REP PHASE =====
        model.eval()
        rep_test.train()  # reps still learn

        # Freeze model params
        for p in model.prior.parameters():
            p.requires_grad_(False)
        for p in model.decoder.parameters():
            p.requires_grad_(False)

        total_test_loss = 0.0
        total_test_recon = 0.0
        total_test_prior = 0.0
        total_test_prior_z = 0.0
        total_test_prior_params = 0.0
        comp_test_sums: dict[str, float] = {}

        for batch_y_test, batch_idx_test in loader_test:
            batch_y_test = batch_y_test.to(device)
            batch_idx_test = batch_idx_test.to(device)

            opt_rep_test.zero_grad()

            (
                loss_test,
                recon_test,
                prior_test,
                recon_comp_test,
                prior_z_test,
                prior_params_test,
            ) = model.batch_loss(
                batch_y_test, rep_layer=rep_test, batch_indices=batch_idx_test
            )

            loss_test.backward()
            opt_rep_test.step()

            bs = batch_y_test.shape[0]
            total_test_loss         += loss_test.item()        * bs
            total_test_recon        += recon_test.item()       * bs
            total_test_prior        += prior_test.item()       * bs
            total_test_prior_z      += prior_z_test.item()     * bs
            total_test_prior_params += prior_params_test.item()* bs

            for name, nll_tensor in recon_comp_test.items():
                comp_test_sums[name] = comp_test_sums.get(name, 0.0) + nll_tensor.item() * bs

        #opt_rep_test.step()
        

        avg_test_loss         = total_test_loss         / n_test
        avg_test_recon        = total_test_recon        / n_test
        avg_test_prior        = total_test_prior        / n_test
        avg_test_prior_z      = total_test_prior_z      / n_test
        avg_test_prior_params = total_test_prior_params / n_test

        test_comp_avgs = {name: s / n_test for name, s in comp_test_sums.items()}

        history["test_loss"].append(avg_test_loss)
        history["test_recon"].append(avg_test_recon)
        history["test_prior"].append(avg_test_prior)
        history["test_prior_z"].append(avg_test_prior_z)
        history["test_prior_params"].append(avg_test_prior_params)
        history["test_recon_components"].append(test_comp_avgs)

        # Unfreeze model params
        for p in model.prior.parameters():
            p.requires_grad_(True)
        for p in model.decoder.parameters():
            p.requires_grad_(True)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={avg_loss:.4f} "
            f"(recon={avg_recon:.4f}, prior={avg_prior:.4f} "
            f"[z={avg_prior_z:.4f}, params={avg_prior_params:.4f}]) | "
            f"test loss={avg_test_loss:.4f} "
            f"(recon={avg_test_recon:.4f}, prior={avg_test_prior:.4f} "
            f"[z={avg_test_prior_z:.4f}, params={avg_test_prior_params:.4f}])"
        )

    # Attach reps & preprocessing to model
    model.rep_train = rep_train
    model.rep_test = rep_test
    model.preproc_mean = base_train.feature_mean
    model.preproc_std  = base_train.feature_std
    model.log_transform = base_train.log_transform
    model.eps = base_train.eps

    return model, history

