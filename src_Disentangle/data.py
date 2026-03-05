import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(
        self,
        X,
        log_transform=False,
        eps=1e-6,
        standardize=False,
        mean=None,
        std=None,
    ):
        """
        General tabular dataset.

        X: numpy array or tensor (n_samples, n_features)
        log_transform: if True, apply log(x + eps)
        standardize: if True, subtract mean and divide by std per feature.
        mean, std: optional (for test data, reuse train stats).
        """
        if isinstance(X, torch.Tensor):
            X = X.clone().detach()
        else:
            X = torch.tensor(X, dtype=torch.float32)

        self.log_transform = log_transform
        self.eps = eps
        self.standardize = standardize

        if log_transform:
            X = torch.log(X + eps)

        if standardize:
            if mean is None or std is None:
                mean = X.mean(dim=0, keepdim=True)
                std = X.std(dim=0, keepdim=True)
            std = std.clone()
            std[std == 0] = 1.0
            X = (X - mean) / std
            self.feature_mean = mean
            self.feature_std = std
        else:
            self.feature_mean = mean
            self.feature_std = std

        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]
