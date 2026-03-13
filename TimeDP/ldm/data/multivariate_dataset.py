# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Multivariate Time Series Generation Dataset for Multivariate TimeDP.

Parallels the existing TSGDataset / TSGDataModule but keeps variates
grouped together as (N, T, C) rather than flattening to (N*C, T, 1).

Each sample returns:
    'context': (T, C) — multivariate time series window
    'data_key': int   — domain index (same semantics as univariate version)

This allows the multivariate training loop to:
    1. Run the frozen PAM independently on each variate
    2. Stack the per-variate masks into M ∈ (B, C, Np)
    3. Pass through the CrossVariateAdapter and CopulaAdapter
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from statsmodels.distributions.empirical_distribution import ECDF
from einops import rearrange


class MultivariateDataset(Dataset):
    """
    Multivariate time series generation dataset.

    Args:
        data_dict: {name: np.ndarray of shape (N, T, C)} where C > 1.
    """

    def __init__(self, data_dict: dict):
        for key, data in data_dict.items():
            assert data.ndim == 3, f"Data must be 3D (N, T, C), but {key} is {data.ndim}D."
            assert data.shape[2] > 1, (
                f"Use standard TSGDataset for univariate data. "
                f"{key} has {data.shape[2]} channels."
            )
        self.data_dict = data_dict
        self._build_index()

    def _build_index(self):
        total = 0
        n_items = {}
        key_list = []
        key_idx_list = []

        for key, data in self.data_dict.items():
            num = data.shape[0]
            total += num
            n_items[key] = num
            key_list.append(key)
            key_idx_list.append(total)

        self.total_items = total
        self.items_dict = n_items
        self.key_list = key_list
        self.key_idx_list = np.array(key_idx_list)

    def get_reweight_sampler(self):
        """Domain-balanced sampler (same logic as univariate TSGDataset)."""
        weights = np.array(
            [1.0 / self.items_dict[k] for k in self.key_list], dtype=np.float32
        )
        sample_weights = np.repeat(
            weights, [self.items_dict[k] for k in self.key_list]
        )
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=self.total_items,
            replacement=True,
        )

    def __len__(self):
        return self.total_items

    def __getitem__(self, idx):
        assert idx < self.total_items
        data_key = int(np.where(self.key_idx_list > idx)[0].min())
        start = self.key_idx_list[data_key - 1] if data_key > 0 else 0
        local_idx = idx - start

        data = self.data_dict[self.key_list[data_key]]
        context = data[local_idx]  # (T, C)

        return {
            'context': context,   # shape: (T, C) — multivariate
            'data_key': data_key,
        }


class MultivariateDataModule(pl.LightningDataModule):
    """
    Data module for multivariate time series generation.

    Key difference from TSGDataModule: normalizes per-variate (each channel
    gets its own normalizer) and preserves the (N, T, C) grouping.

    Args:
        data_path_dict: {name: path} for each dataset.
        n_variates: Number of variates C to use (truncates if data has more).
        window: Sequence length T.
        val_portion: Fraction of data held out for validation.
        normalize: Normalization strategy (same options as TSGDataModule).
        batch_size, num_workers, etc.: Standard DataLoader args.
        reweight: Whether to use domain-balanced sampling.
    """

    def __init__(
        self,
        data_path_dict,
        n_variates=None,
        window=96,
        val_portion=0.1,
        normalize="centered_pit",
        batch_size=128,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        reweight=False,
        **kwargs,
    ):
        super().__init__()
        self.data_path_dict = data_path_dict
        self.n_variates = n_variates
        self.window = window
        self.val_portion = val_portion
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.reweight = reweight
        self.kwargs = kwargs

        # Storage
        self.data_dict = {}
        self.normalizer_dict = {}  # {name: [normalizer_per_variate]}
        self.norm_train_dict = {}
        self.norm_val_dict = {}
        self.key_list = []
        self.actual_n_variates = None

    def prepare_data(self):
        print(f"[Multivariate] Normalizing with: {self.normalize}")
        self.key_list = []

        for data_name, data_path in self.data_path_dict.items():
            self.key_list.append(data_name)
            raw = self._load_multivariate(data_path)  # (N_total, T_full) or (T_full, C)

            # Ensure shape is (N_samples, T, C) via sliding window
            windowed = self._sliding_window(raw)  # (N, T, C)

            # Truncate variates if requested
            if self.n_variates is not None:
                windowed = windowed[:, :, :self.n_variates]

            if self.actual_n_variates is None:
                self.actual_n_variates = windowed.shape[2]
            else:
                assert windowed.shape[2] == self.actual_n_variates, (
                    f"All datasets must have the same number of variates. "
                    f"Expected {self.actual_n_variates}, got {windowed.shape[2]} for {data_name}."
                )

            # Per-variate normalization
            normalizers = []
            norm_data = np.zeros_like(windowed)
            for c in range(windowed.shape[2]):
                variate_data = windowed[:, :, c:c+1]  # (N, T, 1)
                normalizer = self._fit_normalizer(variate_data.flatten())
                normalizers.append(normalizer)
                norm_data[:, :, c] = self._transform(
                    windowed[:, :, c], normalizer
                )

            self.data_dict[data_name] = windowed
            self.normalizer_dict[data_name] = normalizers

            # Split train/val
            train_data, val_data = self._split(norm_data)
            self.norm_train_dict[data_name] = train_data
            self.norm_val_dict[data_name] = val_data

            print(
                f"  {data_name}: {windowed.shape[2]} variates, "
                f"train={train_data.shape}, val={val_data.shape}"
            )

    def _load_multivariate(self, path):
        """Load data and return as 2D array (T_total, C)."""
        if path.endswith(".npy"):
            data = np.load(path).astype(np.float32)
            # Handle various shapes: (N, T), (T, C), (N, T, C)
            if data.ndim == 3:
                # Already (N, T, C) — flatten to (N*T_per_sample, C) for re-windowing
                # Actually just return as-is and handle in sliding_window
                return data
            elif data.ndim == 2:
                return data  # (T, C) or (N, T) — handled by sliding_window
            else:
                raise ValueError(f"Unsupported .npy shape: {data.shape}")
        elif path.endswith(".csv"):
            df = pd.read_csv(path)
            return df.values.astype(np.float32)  # (T, C)
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def _sliding_window(self, data):
        """
        Apply non-overlapping sliding window to produce (N, T, C) samples.

        Args:
            data: Either (T_total, C) or already (N, T, C).

        Returns:
            windowed: (N_windows, T, C)
        """
        if data.ndim == 3:
            # Already windowed — just verify/truncate length
            if data.shape[1] == self.window:
                return data
            elif data.shape[1] > self.window:
                # Re-window: flatten then re-slice
                C = data.shape[2]
                flat = rearrange(data, 'n t c -> (n t) c')
                return self._window_2d(flat)
            else:
                raise ValueError(
                    f"Data window {data.shape[1]} < requested {self.window}"
                )

        # 2D case: (T_total, C)
        return self._window_2d(data)

    def _window_2d(self, data):
        """Slice (T_total, C) into non-overlapping (N, T, C) windows."""
        T_total, C = data.shape
        n_windows = T_total // self.window
        if n_windows == 0:
            raise ValueError(
                f"Data length {T_total} < window {self.window}"
            )
        # Truncate to exact multiple
        data = data[:n_windows * self.window]
        windowed = data.reshape(n_windows, self.window, C)
        return windowed

    def _fit_normalizer(self, data_flat):
        """Fit normalizer on flattened variate data (same logic as TSGDataModule)."""
        normalizer = {}
        if self.normalize == 'zscore':
            normalizer['mean'] = np.nanmean(data_flat)
            normalizer['std'] = np.nanstd(data_flat)
        elif self.normalize == 'minmax':
            normalizer['min'] = np.nanmin(data_flat)
            normalizer['max'] = np.nanmax(data_flat)
        elif self.normalize in ('pit', 'centered_pit'):
            normalizer['ecdf'] = ECDF(data_flat)
        elif self.normalize == 'robust_iqr':
            normalizer['median'] = np.median(data_flat)
            normalizer['iqr'] = np.subtract(*np.percentile(data_flat, [75, 25]))
        elif self.normalize == 'robust_mad':
            normalizer['median'] = np.median(data_flat)
            normalizer['mad'] = np.median(np.abs(data_flat - normalizer['median']))
        return normalizer

    def _transform(self, data, normalizer):
        """Apply normalization (same formulas as TSGDataModule)."""
        if self.normalize == 'zscore':
            return (data - normalizer['mean']) / (normalizer['std'] + 1e-8)
        elif self.normalize == 'minmax':
            return (data - normalizer['min']) / (normalizer['max'] - normalizer['min'] + 1e-8)
        elif self.normalize in ('pit', 'centered_pit'):
            shape = data.shape
            result = normalizer['ecdf'](data.flatten()).reshape(shape)
            if self.normalize == 'centered_pit':
                result = result * 2 - 1
            return result
        elif self.normalize == 'robust_iqr':
            return (data - normalizer['median']) / (normalizer['iqr'] + 1e-8)
        elif self.normalize == 'robust_mad':
            return (data - normalizer['median']) / (normalizer['mad'] + 1e-8)

    def inverse_transform(self, data, data_name, variate_idx):
        """
        Inverse-transform a single variate back to original scale.

        Args:
            data: np.ndarray to inverse-transform.
            data_name: Dataset name (key into normalizer_dict).
            variate_idx: Which variate's normalizer to use.
        """
        normalizer = self.normalizer_dict[data_name][variate_idx]
        if self.normalize == 'zscore':
            return data * normalizer['std'] + normalizer['mean']
        elif self.normalize == 'minmax':
            return data * (normalizer['max'] - normalizer['min']) + normalizer['min']
        elif self.normalize in ('pit', 'centered_pit'):
            ecdf = normalizer['ecdf']
            ecdf.x[0] = ecdf.x[1]
            if self.normalize == 'centered_pit':
                data = (data + 1) / 2
            return np.interp(data, ecdf.y, ecdf.x)
        elif self.normalize == 'robust_iqr':
            return data * normalizer['iqr'] + normalizer['median']
        elif self.normalize == 'robust_mad':
            return data * normalizer['mad'] + normalizer['median']

    def _split(self, data):
        """Shuffle and split into train/val."""
        np.random.shuffle(data)
        n_val = int(data.shape[0] * self.val_portion)
        n_val = max(n_val, 1)  # ensure at least 1 val sample
        return data[:-n_val], data[-n_val:]

    def train_dataloader(self):
        dataset = MultivariateDataset(self.norm_train_dict)
        if self.reweight:
            sampler = dataset.get_reweight_sampler()
            return DataLoader(
                dataset, batch_size=self.batch_size,
                num_workers=self.num_workers, pin_memory=self.pin_memory,
                drop_last=self.drop_last, sampler=sampler, **self.kwargs,
            )
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            drop_last=self.drop_last, **self.kwargs,
        )

    def val_dataloader(self):
        dataset = MultivariateDataset(self.norm_val_dict)
        return DataLoader(
            dataset, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            **self.kwargs,
        )