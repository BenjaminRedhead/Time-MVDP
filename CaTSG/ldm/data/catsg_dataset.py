# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from torch.utils.data import Dataset
import torch
import random

class CaTSGDataset(Dataset):
    def __init__(self, x_path, c_path, window=96, batch_size=None, normalize=False, norm_type='raw',
                 dataset_name=None, split_method=None):
        self.x = np.load(x_path) 
        self.c = np.load(c_path)  
        self.window = window
        self.batch_size = batch_size
        self.normalize = False  
        self.norm_type = 'raw'  
        self.dataset_name = dataset_name
        self.split_method = split_method
            
        self.length = self.x.shape[0]
        self.x_scaler = None
        self.c_scaler = None
        
        # Initialize categorical feature mapping
        self.categorical_indices = set()
        
        # Multi-view augmentation parameters
        self.enable_multi_view = False  # Will be set externally when needed
        self.jitter_sigma = 0.01
        self.scaling_sigma = 0.1
        self.mask_max_frac = 0.03
    
    def _jitter_augment_numerical(self, data, sigma=0.01):
        """Add jitter noise to numerical features only"""
        noise = np.random.normal(0, sigma, data.shape).astype(np.float32)
        return data + noise
    
    def _scaling_augment_numerical(self, data, sigma=0.1):
        """Apply amplitude scaling to numerical features only"""
        scaling_factor = np.random.normal(1.0, sigma, (data.shape[0], 1, data.shape[2])).astype(np.float32)
        return data * scaling_factor
    
    def _mask_augment_all(self, data, max_frac=0.03):
        """Mask random points or segments for all features (time-aligned for categorical)"""
        data_aug = data.copy()
        B, T, D = data.shape
        
        for b in range(B):
            for d in range(D):
                if d in self.categorical_indices:
                    # For categorical: only time-aligned masking (set to padding token 0)
                    if random.random() < 0.5:
                        # Mask random points
                        n_mask = int(T * max_frac * random.random())
                        mask_indices = np.random.choice(T, n_mask, replace=False)
                        data_aug[b, mask_indices, d] = 0.0
                    else:
                        # Mask short segments
                        max_seg_len = max(1, int(T * max_frac))
                        seg_len = random.randint(1, max_seg_len)
                        start_idx = random.randint(0, max(0, T - seg_len))
                        data_aug[b, start_idx:start_idx+seg_len, d] = 0.0
                else:
                    # For numerical: same masking as before
                    if random.random() < 0.5:
                        # Mask random points
                        n_mask = int(T * max_frac * random.random())
                        mask_indices = np.random.choice(T, n_mask, replace=False)
                        data_aug[b, mask_indices, d] = 0.0
                    else:
                        # Mask short segments
                        max_seg_len = max(1, int(T * max_frac))
                        seg_len = random.randint(1, max_seg_len)
                        start_idx = random.randint(0, max(0, T - seg_len))
                        data_aug[b, start_idx:start_idx+seg_len, d] = 0.0
        
        return data_aug
    
    def _apply_augmentations(self, data):
        """Apply random combination of augmentations with differentiated handling for categorical vs numerical"""
        data_aug = data.copy()
        B, T, D = data.shape
        
        # Apply numerical-only augmentations (jitter and scaling) to numerical features
        if random.random() < 0.7:  # 70% chance to apply jitter to numerical features
            num_mask = np.ones((B, T, D), dtype=bool)
            for cat_idx in self.categorical_indices:
                num_mask[:, :, cat_idx] = False
            
            if np.any(num_mask):
                jitter_data = self._jitter_augment_numerical(data_aug, self.jitter_sigma)
                data_aug = np.where(num_mask, jitter_data, data_aug)
        
        if random.random() < 0.5:  # 50% chance to apply scaling to numerical features  
            num_mask = np.ones((B, T, D), dtype=bool)
            for cat_idx in self.categorical_indices:
                num_mask[:, :, cat_idx] = False
            
            if np.any(num_mask):
                scaled_data = self._scaling_augment_numerical(data_aug, self.scaling_sigma)
                data_aug = np.where(num_mask, scaled_data, data_aug)
        
        # Apply masking to all features (time-aligned for both categorical and numerical)
        if random.random() < 0.3:  # 30% chance to apply masking
            data_aug = self._mask_augment_all(data_aug, self.mask_max_frac)
        
        return data_aug
    
    def _apply_synchronized_augmentations(self, x_data, c_data, mode="A"):
        # x_data: (B,T,1), c_data: (B,T,C)
        B, T, _ = x_data.shape
        C = c_data.shape[-1]

        # Determine augmentation parameters
        if mode == "A":
            do_jitter, sigma = True, self.jitter_sigma       # e.g., 0.02
            do_scaling, alpha = True, 0.10                   # U[0.9,1.1]
            do_mask,    p_mask = True, self.mask_max_frac    # e.g., 0.05
        else:  # "B"
            do_jitter, sigma = True, max(1e-6, 0.5*self.jitter_sigma)
            do_scaling, alpha = True, 0.05
            do_mask,    p_mask = True, self.mask_max_frac

        x_aug = x_data.copy()
        c_aug = c_data.copy()

        if do_jitter:
            noise = np.random.normal(0.0, sigma, size=(B, T, 1)).astype(np.float32)
            x_aug = x_aug + noise
            
            # For conditioning, only apply to numerical features
            for c_idx in range(C):
                if c_idx not in self.categorical_indices:
                    c_aug[:, :, c_idx] = c_aug[:, :, c_idx] + noise[:, :, 0]

        if do_scaling:
            s = np.random.uniform(1.0 - alpha, 1.0 + alpha, size=(B, 1, 1)).astype(np.float32)
            x_aug = x_aug * s
            
            # For conditioning, only apply to numerical features
            for c_idx in range(C):
                if c_idx not in self.categorical_indices:
                    c_aug[:, :, c_idx] = c_aug[:, :, c_idx] * s[:, :, 0]

        if do_mask and p_mask > 0:
            mask_idx = self._get_mask_indices((B, T, 1), p_mask)
            x_aug = self._apply_mask_indices(x_aug, mask_idx)
            c_aug = self._apply_mask_indices(c_aug, mask_idx)

        return x_aug, c_aug
    
    def _get_mask_indices(self, data_shape, max_frac):
        """Generate mask indices for synchronized masking"""
        B, T, D = data_shape
        mask_indices = []
        
        for b in range(B):
            # Random masking strategy: either points or segments
            if random.random() < 0.5:
                # Mask random points
                n_mask = int(T * max_frac * random.random())
                points = np.random.choice(T, n_mask, replace=False)
                mask_indices.append(('points', b, points))
            else:
                # Mask short segments
                max_seg_len = max(1, int(T * max_frac))
                seg_len = random.randint(1, max_seg_len)
                start_idx = random.randint(0, max(0, T - seg_len))
                mask_indices.append(('segment', b, start_idx, seg_len))
        
        return mask_indices
    
    def _apply_mask_indices(self, data, mask_indices):
        """Apply pre-computed mask indices to data"""
        data_masked = data.copy()
        
        for mask_info in mask_indices:
            if mask_info[0] == 'points':
                _, b, points = mask_info
                data_masked[b, points, :] = 0.0
            elif mask_info[0] == 'segment':
                _, b, start_idx, seg_len = mask_info
                data_masked[b, start_idx:start_idx+seg_len, :] = 0.0
        
        return data_masked
    
    def transform(self, data):
        """For compatibility with test_prepare"""
        return self
    
    def inverse_transform(self, x_norm, data_name=None):
        """Inverse transform generated samples"""
        if isinstance(x_norm, torch.Tensor):
            x_norm = x_norm.detach().cpu().numpy()
        
        # Handle different input shapes
        if len(x_norm.shape) == 3:  # (B, T, D) -> (B, D, T)
            x_norm = x_norm.transpose(0, 2, 1)
        
        if len(x_norm.shape) == 3:  # (B, D, T)
            B, D, T = x_norm.shape
            x_norm_reshaped = x_norm.transpose(0, 2, 1)  # (B, T, D)
        else:
            x_norm_reshaped = x_norm
            
        x_denorm = self._denormalize_x(x_norm_reshaped)
        return x_denorm

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.x[idx]  # (T, 1)
        c = self.c[idx]  # (T, 4)
        
        result = {
            'x': torch.from_numpy(x).float(),
            'c': torch.from_numpy(c).float()
        }
        
        if self.enable_multi_view:
            x_b = x[np.newaxis, ...]  # (1,T,1)
            c_b = c[np.newaxis, ...]  # (1,T,C)

            x_view1, c_view1 = self._apply_synchronized_augmentations(x_b, c_b, mode="A")
            x_view2, c_view2 = self._apply_synchronized_augmentations(x_b, c_b, mode="B")

            result.update({
                "x_view1": torch.from_numpy(x_view1[0]).float(),
                "c_view1": torch.from_numpy(c_view1[0]).float(),
                "x_view2": torch.from_numpy(x_view2[0]).float(),
                "c_view2": torch.from_numpy(c_view2[0]).float(),
            })
        
        return result