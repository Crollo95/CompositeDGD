# src/model.py
import torch
import torch.nn as nn
from .latent import GaussianMixturePrior, RepresentationLayer
from .decoder import Decoder
from .output_modules import CompositeOutput


class DGDModel(nn.Module):
    def __init__(
        self,
        n_features,
        latent_dim=16,
        n_components=8,
        hidden_dims=(64, 64),
        output_module: str = "gaussian",
        output_module_options: dict | None = None,
        weights_prior_options: dict | None = None,
        dropout_p: float = 0.0,
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device(device)

        self.prior = GaussianMixturePrior(
            latent_dim=latent_dim,
            n_components=n_components,
            weights_prior_options=weights_prior_options,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            out_dim=n_features,
            output_module=output_module,
            output_module_options=output_module_options,
            dropout_p=dropout_p
        )

        self.to(self.device)

    
    def batch_loss(self, batch_y, rep_layer, batch_indices):
        """
        batch_y: (batch, n_features)
        rep_layer: RepresentationLayer (rep_train or rep_test)
        batch_indices: (batch,) LongTensor
    
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
    
        # Latent codes for this batch
        z = rep_layer(batch_indices)
    
        # Decoder forward
        out = self.decoder(z)
        if isinstance(out, (list, tuple)):
            out_params = out
        else:
            out_params = (out,)
    
        # Reconstruction loss (overall)
        recon_loss = self.decoder.negative_log_likelihood(batch_y, *out_params)
    
        # Per-module/component losses
        # (for simple heads, this is just {"GaussianOutput": recon_loss}, etc.)
        recon_components = self.decoder.out.component_losses(batch_y, *out_params)
    
        # Prior: split into z term and parameter prior term
        prior_z_nll = self.prior.z_nll(z)
        prior_param_loss = self.prior.param_prior_loss(z.shape[0])
        prior_loss = prior_z_nll + prior_param_loss
    
        total = recon_loss + prior_loss
    
        return total, recon_loss, prior_loss, recon_components, prior_z_nll, prior_param_loss
        
        

    def get_representations(self, Y, n_steps=200, lr=1e-2):
        Y = Y.to(self.device)
        n_new, _ = Y.shape

        rep_new = RepresentationLayer(n_samples=n_new, latent_dim=self.prior.latent_dim)
        rep_new.to(self.device)

        opt_rep_new = torch.optim.Adam(rep_new.parameters(), lr=lr)
        indices = torch.arange(n_new, device=self.device, dtype=torch.long)

        for _ in range(n_steps):
            opt_rep_new.zero_grad()
            loss, *_ = self.batch_loss(Y, rep_layer=rep_new, batch_indices=indices)
            loss.backward()
            opt_rep_new.step()

        return rep_new.values.detach()

