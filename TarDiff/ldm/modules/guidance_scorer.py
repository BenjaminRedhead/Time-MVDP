# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import time
from scipy.stats import truncnorm  
import math

import torch.nn.functional as F

import numpy as np


class GradDotCalculatorformer:

    def __init__(self,
                 model,
                 train_loader,
                 criterion,
                 device='cuda',
                 normalize='l2',
                 gd_scale=1,
                 mmd_scale=1,
                 pos=True,
                 neg=True):
        """
        Initialize calculator with cached training gradients sum
        
        Args:
            model: trained PyTorch model
            train_loader: DataLoader containing training data
            criterion: loss function
            device: computation device
        """
        self.model = model
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        self.criterion.to(device)
        self.normalize = normalize
        # Cache the sum of training gradients
        self.train_loader = train_loader
        self.positive_guidance = pos
        self.negative_guidance = neg
        self.cached_train_grads = self._compute_train_grad_sum(train_loader)
        self.gd_scale = gd_scale
        self.mmd_scale = mmd_scale

    def compute_mmd_grad(self,
                         x: torch.Tensor,
                         kernel='linear') -> torch.Tensor:

        y = torch.stack([x[0] for x in self.train_loader]).to(self.device)
        x_flat = x.reshape(x.size(0), -1)  # (B1, D*T)
        y_flat = y.reshape(y.size(0), -1).float()  # (B2, D*T)
        if kernel == 'linear':
            K_xx = torch.mm(x_flat, x_flat.t())  # (B1, B1)
            K_yy = torch.mm(y_flat, y_flat.t())  # (B2, B2)
            K_xy = torch.mm(x_flat, y_flat.t())  # (B1, B2)
        elif kernel == 'rbf':
            gamma = 1.0 / x_flat.size(-1)  # 带宽参数
            pairwise_xx = torch.cdist(x_flat, x_flat, p=2)  # (B1, B1)
            K_xx = torch.exp(-gamma * pairwise_xx**2)
            pairwise_yy = torch.cdist(y_flat, y_flat, p=2)  # (B2, B2)
            K_yy = torch.exp(-gamma * pairwise_yy**2)
            pairwise_xy = torch.cdist(x_flat, y_flat, p=2)  # (B1, B2)
            K_xy = torch.exp(-gamma * pairwise_xy**2)
        else:
            raise ValueError("Unsupported kernel type")

        m = x_flat.size(0)
        n = y_flat.size(0)
        mmd = (K_xx.mean() + K_yy.mean() - 2 * K_xy.mean())
        mmd_grad = torch.autograd.grad(mmd, x)

        return self._normalize_gradients(mmd_grad)

    def _normalize_gradients(self, grads):
        """
        Normalize gradients using specified method
        """
        if self.normalize == 'l2':
            # Compute total L2 norm across all parameters
            total_norm = torch.sqrt(sum((g**2).sum() for g in grads))
            return [g / (total_norm + 1e-6) for g in grads]
        elif self.normalize == 'l1':
            # Compute total L1 norm across all parameters
            total_norm = sum(g.abs().sum() for g in grads)
            return [g / (total_norm + 1e-6) for g in grads]
        else:
            return grads

    def _compute_train_grad_sum(self, train_loader):
        """Compute and cache the sum of all training samples' gradients"""

        # Initialize gradient sum
        params = list(self.model.parameters())
        total_loss = torch.tensor(
            0.0,
            device=self.device,
        )
        # Accumulate loss for all training samples
        train_loader = DataLoader(train_loader, batch_size=1, shuffle=False)
        for train_inputs, train_labels in tqdm(train_loader):
            if (train_labels == 0 and not self.negative_guidance) or (
                    train_labels == 1 and not self.positive_guidance):
                continue

            train_inputs = train_inputs.to(self.device).to(torch.float32)
            train_labels = train_labels.to(self.device).to(torch.long)

            mask = torch.full((train_inputs.shape[0], train_inputs.shape[1]),
                              True,
                              dtype=bool,
                              device=self.device)
            outputs = self.model(train_inputs, mask, None, None)
            total_loss += self.criterion(outputs, train_labels)

        # Get gradient of total loss
        start_time = time.time()
        grad_sum = torch.autograd.grad(total_loss, params, allow_unused=True)
        filtered = [(p, g) for p, g in zip(params, grad_sum) if g is not None]
        if not filtered:
            raise ValueError("No parameter received gradient!")
        filtered_params, filtered_grads = zip(*filtered)
        print(f"Gradient computation time: {time.time() - start_time:.2f}s")
        torch.cuda.empty_cache()
        return self._normalize_gradients(filtered_grads)
        #return filtered_grads
    def compute_gradient(self, test_sample, test_label):
        """
        Compute gradient of grad-dot using cached training gradients
        
        Args:
            test_sample: single test input tensor
            test_label: ground truth label for test sample
        """
        self.model.eval()
        test_sample = test_sample.transpose(2, 1)
        # Prepare test sample
        test_sample = test_sample.detach().requires_grad_(True)
        test_sample = test_sample.to(self.device)
        #test_label = torch.tensor([test_label]).to(self.device)
        test_label = test_label.to(self.device)
        # Get test gradient
        mask = torch.full((test_sample.shape[0], test_sample.shape[1]),
                          True,
                          dtype=bool,
                          device=self.device)
        test_output = self.model(test_sample, mask, None, None)
        test_loss = self.criterion(test_output, test_label)
        test_grads = torch.autograd.grad(test_loss,
                                         self.model.parameters(),
                                         create_graph=True,
                                         allow_unused=True)
        filtered = [(p, g) for p, g in zip(self.model.parameters(), test_grads)
                    if g is not None]
        if not filtered:
            raise ValueError("No parameter received gradient!")
        filtered_params, filtered_grads = zip(*filtered)
        test_grads = self._normalize_gradients(filtered_grads)
        # Compute single dot product with cached gradients sum
        total_dot = sum(
            (test_g * cached_g).sum()
            for test_g, cached_g in zip(test_grads, self.cached_train_grads))
        # Get gradient w.r.t test_sample
        grad_wrt_sample = torch.autograd.grad(total_dot,
                                              test_sample,
                                              create_graph=False)[0]
        #mmd_grad=self.compute_mmd_grad(test_sample)[0].squeeze(0).transpose(2,1)
        gd_grad = grad_wrt_sample.squeeze(0).transpose(2, 1)

        torch.cuda.empty_cache()
        dynamic_scale = self.gd_scale

        return dynamic_scale * gd_grad

    def compute_influence(self, test_sample, test_label, t):
        """
        Compute gradient of grad-dot using cached training gradients
        
        Args:
            test_sample: single test input tensor
            test_label: ground truth label for test sample
        """
        self.model.eval()

        test_sample = test_sample.transpose(2, 1)
        # Prepare test sample
        test_sample = test_sample.detach().requires_grad_(True)
        test_sample = test_sample.to(self.device)
        #test_label = torch.tensor([test_label]).to(self.device)
        test_label = test_label.to(self.device)
        # Get test gradient
        mask = torch.full((test_sample.shape[0], test_sample.shape[1]),
                          True,
                          dtype=bool,
                          device=self.device)
        test_output = self.model(test_sample, mask, None, None, t_idx=t)
        test_loss = self.criterion(test_output, test_label)
        test_grads = torch.autograd.grad(test_loss,
                                         self.model.parameters(),
                                         create_graph=True,
                                         allow_unused=True)
        filtered = [(p, g) for p, g in zip(self.model.parameters(), test_grads)
                    if g is not None]
        if not filtered:
            raise ValueError("No parameter received gradient!")
        filtered_params, filtered_grads = zip(*filtered)
        test_grads = self._normalize_gradients(filtered_grads)
        # Compute single dot product with cached gradients sum
        total_dot = sum(
            (test_g * cached_g).sum()
            for test_g, cached_g in zip(test_grads, self.cached_train_grads))
        torch.cuda.empty_cache()
        total_dot = total_dot.detach().cpu().numpy()
        return total_dot

    def compute_per_class_grad_norm_stats(self):
        """
        For each sample in the training set:
        1. Compute the gradient of the loss w.r.t. all model parameters;
        2. Collapse each parameter-wise gradient to its L2 norm, then obtain a
            single scalar gradient norm for that sample;
        3. Group these norms by class label and report mean ± std for every class.

        Returns
        -------
        stats : dict
            keys   → class labels
            values → (mean_norm, std_norm)
        """
        self.model.eval()
        grad_norms = {}

        # Iterate one sample at a time so we can attribute a single norm per example.
        data_loader = DataLoader(self.train_loader,
                                 batch_size=1,
                                 shuffle=False)

        for inputs, labels in tqdm(data_loader,
                                   desc="Computing per-class grad norms"):
            inputs = inputs.to(self.device).to(torch.float32)
            labels = labels.to(self.device)

            # Create a full-True mask because the backbone expects it
            mask = torch.full((inputs.shape[0], inputs.shape[1]),
                              True,
                              dtype=torch.bool,
                              device=self.device)

            outputs = self.model(inputs, mask, None, None)
            loss = self.criterion(outputs, labels)

            # Gradients w.r.t. all parameters
            grads = torch.autograd.grad(loss,
                                        self.model.parameters(),
                                        create_graph=False,
                                        allow_unused=True)

            # Discard parameters that produced None (e.g. frozen layers)
            filtered_grads = [g for g in grads if g is not None]
            if not filtered_grads:
                continue

            # L2 norm over all parameters: sqrt(Σ‖g‖₂²)
            grad_norm_sq = sum(torch.sum(g**2) for g in filtered_grads)
            grad_norm = torch.sqrt(grad_norm_sq)

            # Batch size is 1 here, so labels is a scalar
            label_val = labels.item() if labels.numel(
            ) == 1 else labels.tolist()[0]
            grad_norms.setdefault(label_val, []).append(grad_norm.item())

        stats = {}
        for label_val, norm_list in grad_norms.items():
            mean, std = np.mean(norm_list), np.std(norm_list)
            stats[label_val] = (mean, std)
            print(f"Class {label_val}: {mean:.4f} ± {std:.4f}")
        return stats

    def compute_classifier_guidance(self, test_sample, test_label):
        """
        Vectorised computation of ∇ₓ log p(y|x) for a batch.

        Parameters
        ----------
        test_sample : Tensor [B, D, T]
            Raw batch (before transpose). D = channels, T = timesteps.
        test_label  : Tensor [B] or [B, 1]
            Ground-truth class indices.

        Returns
        -------
        guidance_grad : Tensor [B, D, T]
            Input-space gradients scaled by self.gd_scale.
        """
        self.model.eval()

        # Model expects (B, T, D); enable gradient on inputs
        test_sample = (test_sample.transpose(
            2, 1).detach().requires_grad_(True).to(self.device))
        test_label = test_label.to(self.device)
        if test_label.ndim > 1:
            test_label = test_label.squeeze(-1)

        B = test_sample.shape[0]

        mask = torch.full((B, test_sample.shape[1]),
                          True,
                          dtype=torch.bool,
                          device=self.device)
        logits = self.model(test_sample, mask, None, None)  # [B, C]
        log_probs = F.log_softmax(logits, dim=1)  # [B, C]

        # Select log-probability of the true class for each sample
        log_selected = log_probs.gather(1,
                                        test_label.view(-1,
                                                        1)).squeeze(1)  # [B]

        # Sum over batch to obtain a scalar; gradients remain sample-wise
        loss = log_selected.sum()
        grad = torch.autograd.grad(loss,
                                   test_sample,
                                   create_graph=False,
                                   retain_graph=False,
                                   allow_unused=True)[0]

        return grad.transpose(2, 1) * self.gd_scale


class GradDotCalculator:

    def __init__(self,
                 model,
                 train_loader,
                 criterion,
                 gd_scale=1,
                 device='cuda',
                 normalize='l2'):
        """
        Compute and cache the gradient sum over the *entire* training set,
        then provide fast “grad-dot” calculations for test samples.

        Parameters
        ----------
        model : torch.nn.Module
            A pre-trained network (e.g., RNN).
        train_loader : DataLoader
            Loader that iterates over the full training dataset.
        criterion : callable
            Loss function used for both training and evaluation.
        gd_scale : float, default=1
            Optional scaling factor applied to the final input-space gradient.
        device : {'cuda', 'cpu'}, default='cuda'
            Computation device.
        normalize : {'l2', 'l1', None}, default='l2'
            How to normalise parameter-space gradients before dot-product.
        """
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.device = device
        self.normalize = normalize
        self.gd_scale = gd_scale
        self.cached_train_grads = self._compute_train_grad_sum(train_loader)

    def _normalize_gradients(self, grads):
        """
        Apply the requested normalisation to a list of parameter gradients.

        Parameters
        ----------
        grads : list[Tensor]
            One tensor per parameter.

        Returns
        -------
        list[Tensor]
            Normalised gradients (or originals if `normalize` is None).
        """
        if self.normalize == 'l2':
            total_norm = torch.sqrt(sum((g**2).sum() for g in grads))
            return [g / (total_norm + 1e-6) for g in grads]
        elif self.normalize == 'l1':
            total_norm = sum(g.abs().sum() for g in grads)
            return [g / (total_norm + 1e-6) for g in grads]
        else:
            return grads

    def _compute_train_grad_sum(self, train_loader):
        """
        Accumulate the loss over the *whole* training set, then take a single
        backward pass to obtain Σ∇θ L.  This treats every sample as if it were
        in one giant batch.  Beware of memory usage on very large datasets.

        Returns
        -------
        list[Tensor]
            Normalised gradient sum, one tensor per trainable parameter.
        """
        params = list(self.model.parameters())
        total_loss = torch.tensor(0.0, device=self.device)

        for inputs, labels in tqdm(train_loader,
                                   desc="Compute Train Grad Sum"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).to(torch.float32)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            total_loss += loss

        grad_sum = torch.autograd.grad(total_loss, params, allow_unused=True)
        filtered = [(p, g) for p, g in zip(params, grad_sum) if g is not None]
        _, filtered_grads = zip(*filtered)
        grad_sum = self._normalize_gradients(filtered_grads)
        torch.cuda.empty_cache()
        return grad_sum

    def compute_gradient(self, test_sample, test_label):
        """
        Compute the input-space gradient induced by a “grad-dot” score:

            ⟨∇θ L_test,  Σ∇θ L_train⟩

        Steps
        -----
        1. Back-prop through the test loss to obtain ∇θ L_test.
        2. Dot-product with the cached training-set gradient sum.
        3. Back-prop that scalar w.r.t. the *input* to get ∇ₓ score.

        Parameters
        ----------
        test_sample : Tensor [B, D, T]
            Input time-series (channels-first, will be transposed internally).
        test_label : Tensor
            Corresponding ground-truth labels.

        Returns
        -------
        Tensor [B, D, T]
            Input-space gradient scaled by `gd_scale`.
        """
        # Model expects (B, T, D)
        test_sample = test_sample.transpose(2, 1).to(self.device)
        test_sample.requires_grad_(True)

        test_label = test_label.to(self.device).to(torch.float32)

        with torch.backends.cudnn.flags(enabled=False):
            test_output = self.model(test_sample)
        test_loss = self.criterion(test_output, test_label)

        test_grads = torch.autograd.grad(test_loss,
                                         self.model.parameters(),
                                         create_graph=True)

        filtered = [(p, g) for p, g in zip(self.model.parameters(), test_grads)
                    if g is not None]
        _, test_grads = zip(*filtered)
        test_grads = self._normalize_gradients(test_grads)

        total_dot = sum((tg * cg).sum()
                        for tg, cg in zip(test_grads, self.cached_train_grads))

        grad_wrt_sample = torch.autograd.grad(total_dot, test_sample)[0]
        return grad_wrt_sample.transpose(2, 1) * self.gd_scale
