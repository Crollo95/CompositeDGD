import torch
import torch.nn as nn
from .latent import GaussianMixturePrior, RepresentationLayer, SplitRepresentationLayer
from .decoder import Decoder, DecoderCNN
from .output_modules import CompositeOutput, NegativeBinomialOutput


class DGDModel(nn.Module):
    def __init__(
        self,
        n_features,
        n_z: int = 8,
        n_w: int = 8,
        n_components_z: int = 8,
        n_components_w: int = 8,
        beta_w: float = 1.0,
        hidden_dims=(64, 64),
        output_module: str = "gaussian",
        output_module_options: dict | None = None,
        weights_prior_options_z: dict | None = None,
        weights_prior_options_w: dict | None = None,
        dropout_p: float = 0.0,
        
        decoder_type: str = "mlp", # "cnn"
        img_shape: tuple = (1, 28, 28),
        cnn_likelihood: str = "bernoulli",
        film: bool = False,
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        self.n_z = int(n_z)
        self.n_w = int(n_w)
        latent_dim = self.n_z + self.n_w
        self.beta_w = beta_w

        # separate priors
        self.prior_z = GaussianMixturePrior(
            latent_dim=self.n_z,
            n_components=n_components_z,
            weights_prior_options=weights_prior_options_z,
            #init_log_var = -4, ### for MMCA2
            #log_var_prior_options = {"mean": -4.0, "stddev": 0.2}, ### for MMCA2
            #means_prior_options = {"radius": 0.5, "sharpness": 10.0} ### for MMCA2
        )
        self.prior_w = GaussianMixturePrior(
            latent_dim=self.n_w,
            n_components=n_components_w,
            weights_prior_options=weights_prior_options_w,
            #init_log_var = -4, ### for MMCA2
            #log_var_prior_options = {"mean": -4.0, "stddev": 0.2}, ### for MMCA2
            #means_prior_options = {"radius": 0.5, "sharpness": 10.0} ### for MMCA2
        )

        self.decoder_type = decoder_type
        if decoder_type == "cnn":
            self.decoder = DecoderCNN(
                latent_dim=latent_dim,
                img_shape=img_shape,
                likelihood=cnn_likelihood,
                dropout_p=dropout_p,
                film=film,
                n_w=self.n_w,
            )
        else:
            self.decoder = Decoder(
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                out_dim=n_features,
                output_module=output_module,
                output_module_options=output_module_options,
                dropout_p=dropout_p,
            )

        self.to(self.device)

    
    def batch_loss(self, batch_y, rep_layer, batch_indices, cond_mask=None, z_prior_on_y0_only:bool=False, scaling_factor=None):
        """
        batch_y: (batch, n_features)
        rep_layer: RepresentationLayer (rep_train or rep_test)
        batch_indices: (batch,) LongTensor
        scaling_factor: (batch,) pre-computed library size factors (optional).
            If provided and mean_mode='softmax', used directly instead of computing from batch_y.

        returns:
            total_loss,
            recon_loss,
            prior_loss,          # z_nll + param_prior_loss
            recon_components,    # dict[str, scalar_tensor]
            prior_z_nll,         # scalar: -E[log p(z | GMM)]
            prior_param_loss,    # scalar: -log p(parameters) / batch_size
        """
        batch_y = batch_y.to(self.device)
        batch_indices = batch_indices.to(self.device)
        cond_mask = cond_mask.to(self.device)

        # get split reps
        z, w_used = rep_layer(batch_indices, cond_mask=cond_mask)  # (B,n_z), (B,n_w)
        latent = torch.cat([z, w_used], dim=1)                     # (B,n_z+n_w)
    
        # Decoder forward
        out = self.decoder(latent)
        if isinstance(out, (list, tuple)):
            out_params = out
        else:
            out_params = (out,)
    
        # Reconstruction loss (overall)
        if hasattr(self.decoder, "out") and isinstance(self.decoder.out, NegativeBinomialOutput):
            if self.decoder.out.mean_mode == "softmax":
                if scaling_factor is not None:
                    # use pre-computed global scaling factors (preferred)
                    sf_norm = torch.as_tensor(scaling_factor, dtype=batch_y.dtype, device=self.device).view(-1)
                else:
                    # fallback: normalize within batch
                    sf = batch_y.sum(dim=1)
                    sf_norm = sf / (sf.median() + 1e-8)
                scaling_factor = sf_norm
            else:
                scaling_factor = None
            recon_loss = self.decoder.negative_log_likelihood(batch_y, *out_params, scaling_factor=scaling_factor)
        else:
            recon_loss = self.decoder.negative_log_likelihood(batch_y, *out_params)


        # Per-module/component losses
        # (for simple heads, this is just {"GaussianOutput": recon_loss}, etc.)
        if hasattr(self.decoder, "out") and hasattr(self.decoder.out, "component_losses"):
            if isinstance(self.decoder.out, NegativeBinomialOutput) and self.decoder.out.mean_mode == "softmax":
                recon_components = {"recon": recon_loss.detach()}
            else:
                recon_components = self.decoder.out.component_losses(batch_y, *out_params)
        else:
            recon_components = {"recon": recon_loss.detach()}

    
        # =========================
        # Z PRIOR
        # =========================
        cm = cond_mask.view(-1)  # (B,)

        if z_prior_on_y0_only:
            inactive_mask = (cm <= 0.5)  # y=0 samples
            n0 = int(inactive_mask.sum().item())

            # choose subset for PARAMETER updates (avoid empty)
            if n0 == 0:
                z_for_params = z
                n_for_prior = z.shape[0]
            else:
                z_for_params = z[inactive_mask]
                n_for_prior = n0

            # (A) update z embeddings using ALL samples, but freeze GMM params
            prior_z_nll_embed = self.prior_z.z_nll_detached_params(z)

            # (B) update GMM(z) params using ONLY y=0 samples, but freeze z
            prior_z_nll_params = self.prior_z.z_nll(z_for_params.detach())

            # combine
            prior_z_nll = prior_z_nll_embed + prior_z_nll_params

            # parameter prior (params only anyway) — scale by y=0 count like before
            prior_z_params = self.prior_z.param_prior_loss(n_for_prior)

        else:
            # standard behavior: everyone updates from everyone
            prior_z_nll = self.prior_z.z_nll(z)
            prior_z_params = self.prior_z.param_prior_loss(z.shape[0])

    
        # =========================
        # W PRIOR
        # =========================
        active_mask = (cm > 0.5)
        n_w = int(active_mask.sum().item())
    
        if n_w == 0:
            prior_w_nll = torch.zeros((), device=self.device, dtype=z.dtype)
            prior_w_params = torch.zeros((), device=self.device, dtype=z.dtype)
        else:
            w_active = w_used[active_mask]
            prior_w_nll = self.prior_w.z_nll(w_active)
            prior_w_params = self.prior_w.param_prior_loss(n_w)
        
        beta_z, beta_w = 1.0, self.beta_w
        prior_loss = beta_z*(prior_z_nll + prior_z_params) + beta_w*(prior_w_nll + prior_w_params)
        total = recon_loss + prior_loss
    
        return (
            total,
            recon_loss,
            prior_loss,
            recon_components,
            prior_z_nll,
            prior_z_params,
            prior_w_nll,
            prior_w_params,
        )



    def get_representations(self, Y, cond_mask=None, n_steps=200, lr=1e-2):
        """
        Fit a new representation table for new data.
        NOTE: needs a rep_layer class that supports split (z,w) and cond_mask.
        """
        Y = Y.to(self.device)
        n_new, _ = Y.shape
    
        rep_new = SplitRepresentationLayer(n_samples=n_new, n_z=self.n_z, n_w=self.n_w).to(self.device)
        
        if cond_mask is not None:
            cond_mask = torch.as_tensor(cond_mask, device=self.device, dtype=torch.float32)
            with torch.no_grad():
                rep_new.w[cond_mask <= 0.5] = 0.0
    
        opt = torch.optim.Adam(rep_new.parameters(), lr=lr)
        idx = torch.arange(n_new, device=self.device, dtype=torch.long)
    
        if cond_mask is not None:
            cond_mask = torch.as_tensor(cond_mask, device=self.device, dtype=torch.float32)
    
        for _ in range(n_steps):
            opt.zero_grad()
            loss, *_ = self.batch_loss(Y, rep_layer=rep_new, batch_indices=idx, cond_mask=cond_mask)
            loss.backward()
            opt.step()
    
        return torch.cat([rep_new.z.detach(), rep_new.w.detach()], dim=1)
