# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from datetime import datetime
import pandas as pd
from distutils.util import strtobool
from statsmodels.distributions.empirical_distribution import ECDF
from torch.utils.data import WeightedRandomSampler
from einops import rearrange
import os

class TSClassCondDataset(
        Dataset):  # For generation task. Unified Univariate Generation Dataset

    def __init__(self, data_dict: dict):
        for key, data in data_dict.items():
            assert type(
                data
            ) is tuple, f"Data must be tuple, but {key} got {type(data)}."
        self.data_dict = data_dict
        self.data_key, self.data = next(iter(self.data_dict.items()))
        self.total_items = len(self.data[0])
        self.pos_indices = np.where(self.data[1] == 1)[0]
        self.neg_indices = np.where(self.data[1] == 0)[0]

    def __len__(self):
        return self.total_items  # self.num_slices

    def __getitem__(self, idx):
        if idx % 2 == 0:
            sample_idx = np.random.choice(self.pos_indices)
        else:
            sample_idx = np.random.choice(self.neg_indices)
        sample_idx = idx
        context = self.data[0][sample_idx]
        label = self.data[1][sample_idx]

        return {
            'context': context,  # shape: (window,)
            'data_key': 0,
            'label': label
        }


class TSClassCondTrainDataModule(pl.LightningDataModule):
    '''
    Data module for unified time series generation task.
    Slicing is also done with this module. So the train/val is i.i.d within train dataset.
    '''

    def __init__(self,
                 data_path_dict,
                 window=96,
                 val_portion=0.1,
                 as_tensor: bool = True,
                 normalize=None,
                 batch_size=128,
                 num_workers=0,
                 pin_memory=True,
                 drop_last=False,
                 reweight=False,
                 input_dim=1,
                 **kwargs):
        super().__init__()
        self.data_path_dict = data_path_dict  # {data_name: data_path}
        self.data_dict = {}
        self.norm_data_dict = {}
        self.normalizer_dict = {}
        self.norm_train_dict = {}
        self.norm_val_dict = {}
        self.window = window
        self.val_portion = val_portion
        self.as_tensor = as_tensor
        assert normalize in [
            None, 'zscore', 'robust_iqr', 'robust_mad', 'pit', 'centered_pit',
            'minmax'
        ], f"Normalize({normalize}) must be in (zscore, robust_iqr, robust_mad, pit)."
        self.normalize = normalize

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.reweight = reweight
        # self.transform = None
        self.kwargs = kwargs
        self.key_list = []
        self.drop_last = drop_last
        self.input_dim = input_dim

    def prepare_data(self) -> None:
        # loaded_data, *_ = convert_tsf_to_dataframe(self.data_dir)
        # self.data = np.stack(loaded_data['series_value'].values)  # loaded_data.values
        print(f"Normalizing data with: {self.normalize}")
        self.key_list = []
        for data_name, data_path in self.data_path_dict.items():
            self.key_list.append(data_name)
            this_data = load_data_from_file(data_path)
            # if this_data is a tuple, then it is (data, label)
            if type(this_data) is tuple:
                this_data, label = this_data
                this_data = this_data[:, :, :self.window]

                this_data = rearrange(this_data, 'n c t -> n t c')

            if this_data.ndim == 3:  # in shape (N, T, C)
                if self.input_dim == 1:
                    this_data = rearrange(
                        this_data, 'n t c -> (n c) t 1')  # to shape (N*C, T)
                    label = np.repeat(label,
                                      this_data.shape[0] // label.shape[0])

            # first normalize, then split
            normalizer = self.fit_normalizer(this_data)
            self.data_dict[data_name] = this_data
            self.normalizer_dict[data_name] = normalizer
            norm_data = self.transform(this_data, normalizer)
            self.norm_data_dict[data_name] = norm_data
            train_data, val_data = self.split_train_val(
                norm_data, label)  # slice and split here
            guidance_file = 'guidance_tuple.pkl'
            if not os.path.exists(
                    os.path.join(os.path.dirname(data_path), guidance_file)):
                pd.to_pickle((val_data[0].transpose(0, 2, 1), val_data[1]),
                             os.path.join(os.path.dirname(data_path),
                                          guidance_file))

            self.norm_train_dict[data_name] = train_data
            self.norm_val_dict[data_name] = val_data

            print(
                f"Loaded data: {data_name}; Train shape: {train_data[0].shape}, Validation shape: {val_data[0].shape}."
            )
            print(f"With normalizer fit as: {normalizer}")

    def expand_data_for_finetune(self, data_name, data_path):
        self.key_list.append(data_name)
        this_data = load_data_from_file(data_path).astype(np.float32)
        # first normalize, then split
        normalizer = self.fit_normalizer(this_data)
        self.data_dict[data_name] = this_data
        self.normalizer_dict[data_name] = normalizer
        norm_data = self.transform(this_data, normalizer)
        self.norm_data_dict[data_name] = norm_data
        train_data, val_data = self.split_train_val(
            norm_data)  # slice and split here
        self.norm_train_dict[data_name] = train_data
        self.norm_val_dict[data_name] = val_data

        print(
            f"Loaded data: {data_name}; Train shape: {train_data.shape}, Validation shape: {val_data.shape}."
        )
        print(f"With normalizer fit as: {normalizer}")

    def split_train_val(self, data: np.ndarray, label=None):

        indices = np.arange(len(data))
        np.random.shuffle(indices)
        # 按索引打乱 data 和 label
        data = data[indices]
        if label is not None:
            label = label[indices]
            train_data = data[:-int(len(data) * self.val_portion)]
            train_label = label[:-int(len(data) * self.val_portion)]
            val_data = data[-int(len(data) * self.val_portion):]
            val_label = label[-int(len(data) * self.val_portion):]
            return (train_data, train_label), (val_data, val_label)
        else:
            train_data = data[:-int(len(data) * self.val_portion)]
            val_data = data[-int(len(data) * self.val_portion):]
            return (train_data, None), (val_data, None)

    def train_dataloader(self):
        train_dataset = TSClassCondDataset(self.norm_train_dict)
        sampler = None
        if self.reweight:
            sampler = train_dataset.get_reweight_sampler()
            return DataLoader(train_dataset,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              pin_memory=self.pin_memory,
                              drop_last=self.drop_last,
                              sampler=sampler,
                              **self.kwargs)
        else:
            return DataLoader(train_dataset,
                              batch_size=self.batch_size,
                              num_workers=self.num_workers,
                              pin_memory=self.pin_memory,
                              shuffle=True,
                              drop_last=self.drop_last,
                              **self.kwargs)

    def val_dataloader(self):
        val_dataset = TSClassCondDataset(self.norm_val_dict)
        return DataLoader(val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          **self.kwargs)

    def test_dataloader(self, **kwargs):
        return None

    def fit_normalizer(self, data: np.ndarray):
        normalizer = {}
        data = data.flatten()
        if self.normalize == 'zscore':
            normalizer['mean'] = np.nanmean(data)
            normalizer['std'] = np.nanstd(data)
        elif self.normalize == 'robust_iqr':
            normalizer['median'] = np.median(data)
            normalizer['iqr'] = np.subtract(*np.percentile(data, [75, 25]))
        elif self.normalize == 'robust_mad':
            normalizer['median'] = np.median(data)
            normalizer['mad'] = np.median(np.abs(data - normalizer['median']))
        elif self.normalize == 'minmax':
            normalizer['min'] = np.nanmin(data)
            normalizer['max'] = np.nanmax(data)
        elif self.normalize == 'pit' or self.normalize == 'centered_pit':
            ecdf = ECDF(data)
            # ecdf.x[0] = np.min(data)
            normalizer['ecdf'] = ecdf
            # normalizer['inverter'] = monotone_fn_inverter(ecdf)
        return normalizer

    def transform(self, data: np.ndarray, normalizer=None, data_name=None):
        # if data_name is specified, the normalizer argument will be ignored.
        assert normalizer is not None or data_name is not None, "Must specify either normalizer or data name."
        if data_name is not None:
            assert data_name in self.normalizer_dict.keys(
            ), f"Data name({data_name}) must be in normalizer dict key."
            normalizer = self.normalizer_dict[data_name]
        if self.normalize == 'zscore':
            return (data - normalizer['mean']) / (normalizer['std'] + 1e-8)
        elif self.normalize == 'robust_iqr':
            return (data - normalizer['median']) / (normalizer['iqr'] + 1e-8)
        elif self.normalize == 'robust_mad':
            return (data - normalizer['median']) / (normalizer['mad'] + 1e-8)
        if self.normalize == 'minmax':
            return (data - normalizer['min']) / (normalizer['max'] -
                                                 normalizer['min'] + 1e-8)
        elif self.normalize == 'pit' or self.normalize == 'centered_pit':
            data_shape = data.shape
            norm_data = normalizer['ecdf'](data.flatten()).reshape(data_shape)
            if self.normalize == 'centered_pit':
                norm_data = norm_data * 2 - 1
            return norm_data

    def inverse_transform(self,
                          data: np.ndarray,
                          normalizer=None,
                          data_name=None):
        # if data_name is specified, the normalizer argument will be ignored.
        assert normalizer is not None or data_name is not None, "Must specify either normalizer or data name."
        if data_name is not None:
            assert data_name in self.normalizer_dict.keys(
            ), f"Data name({data_name}) must be in normalizer dict key."
            normalizer = self.normalizer_dict[data_name]
        if self.normalize == 'zscore':
            return data * normalizer['std'] + normalizer['mean']
        elif self.normalize == 'robust_iqr':
            return data * normalizer['iqr'] + normalizer['median']
        elif self.normalize == 'robust_mad':
            return data * normalizer['mad'] + normalizer['median']
        if self.normalize == 'minmax':
            return data * (normalizer['max'] -
                           normalizer['min']) + normalizer['min']
        elif self.normalize == 'pit' or self.normalize == 'centered_pit':
            ecdf: ECDF = normalizer['ecdf']
            ecdf.x[0] = ecdf.x[1]
            if self.normalize == 'centered_pit':
                data = (data + 1) / 2
            return np.interp(data, ecdf.y, ecdf.x)


class TSGDataset(Dataset):  # For generation task. Unified Univariate Generation Dataset
    def __init__(self, data_dict: dict):
        for key, data in data_dict.items():
            assert data.ndim == 3, f"Data must be 3D, but {key} got {data.ndim}D."
        self.data_dict = data_dict
        self.cal_data_stats()
        
    def cal_data_stats(self):
        total_items = 0
        n_items_dict = {}
        key_list = []
        key_idx_list = []
        
        for key, data in self.data_dict.items():
            num_items = data.shape[0]
            total_items += num_items
            n_items_dict[key] = num_items
            key_list.append(key)
            key_idx_list.append(total_items)
            
        self.total_items = total_items
        self.items_dict = n_items_dict    
        self.key_list = key_list
        self.key_idx_list = np.array(key_idx_list)
    
    def get_reweight_sampler(self):
        dataset_weights = np.array([1 / self.items_dict[key] for key in self.key_list], dtype=np.float32)
        sample_weights = np.repeat(dataset_weights, [self.items_dict[key] for key in self.key_list])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=self.total_items, replacement=True)
        return sampler
        
    def __len__(self):
        return self.total_items  # self.num_slices
    
    def __getitem__(self, idx):
        assert idx < self.total_items, f"Index({idx}) must be less than number of items({self.total_items})."
        data_key = np.where(self.key_idx_list > idx)[0].min()  # np.argmin(self.key_idx_list > idx)
        data_start_idx = self.key_idx_list[data_key-1] if data_key > 0 else 0
        data: np.ndarray = self.data_dict[self.key_list[data_key]]
        
        valid_idx = idx  - data_start_idx
        context = data[valid_idx,:,0]

        return {
            'context': context,  # shape: (window,)
            'data_key': data_key
            }  

    
class TSGDataModule(pl.LightningDataModule):
    '''
    Data module for unified time series generation task.
    Slicing is also done with this module. So the train/val is i.i.d within train dataset.
    '''
    def __init__(self, data_path_dict, window=96, val_portion=0.1, as_tensor:bool=True, normalize="centered_pit", batch_size=128, num_workers=0, pin_memory=True, drop_last=False, reweight=False, **kwargs):
        super().__init__()
        self.data_path_dict = data_path_dict  # {data_name: data_path}
        self.data_dict = {}
        self.norm_data_dict = {}
        self.normalizer_dict = {}
        self.norm_train_dict = {}
        self.norm_val_dict = {}
        self.window = window
        self.val_portion = val_portion
        self.as_tensor = as_tensor
        assert normalize in [None, 'zscore', 'robust_iqr', 'robust_mad', 'pit', 'centered_pit', 'minmax'], f"Normalize({normalize}) must be in (zscore, robust_iqr, robust_mad, pit)."
        self.normalize = normalize
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.reweight = reweight
        # self.transform = None
        self.kwargs = kwargs
        self.key_list = []
        self.drop_last = drop_last
        
    def prepare_data(self) -> None:
        print(f"Normalizing data with: {self.normalize}")
        self.key_list = []
        for data_name, data_path in self.data_path_dict.items():
            self.key_list.append(data_name)
            this_data = load_data_from_file(data_path).astype(np.float32)
            if this_data.ndim == 3:  # in shape (N, T, C)
                this_data = rearrange(this_data, 'n t c -> (n c) t 1')  # to shape (N*C, T)
            # first normalize, then split
            normalizer = self.fit_normalizer(this_data)
            self.data_dict[data_name] = this_data
            self.normalizer_dict[data_name] = normalizer
            norm_data = self.transform(this_data, normalizer)
            self.norm_data_dict[data_name] = norm_data
            train_data, val_data = self.split_train_val(norm_data)  # slice and split here
            self.norm_train_dict[data_name] = train_data
            self.norm_val_dict[data_name] = val_data
                        
            print(f"Loaded data: {data_name}; Train shape: {train_data.shape}, Validation shape: {val_data.shape}.")
            print(f"With normalizer fit as: {normalizer}")
    
    def split_train_val(self, data: np.ndarray):
        # By default, data are sliced into non-overlapped sequences.
        # shuffle stack_data, only along the first dimension
        np.random.shuffle(data)
        total_instances = data.shape[0]
        num_val_instances = int(total_instances * self.val_portion)
        train_data = data[:-num_val_instances]
        val_data = data[-num_val_instances:]
        
        return train_data, val_data
        
    def train_dataloader(self):
        train_dataset = TSGDataset(self.norm_train_dict)
        sampler = None
        if self.reweight:
            sampler = train_dataset.get_reweight_sampler()
            return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last, sampler=sampler, **self.kwargs)
        else:
            return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True, drop_last=self.drop_last, **self.kwargs)
    
    def val_dataloader(self):
        val_dataset = TSGDataset(self.norm_val_dict)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, **self.kwargs)
    
    def test_dataloader(self, **kwargs):
        return None
       
    def fit_normalizer(self, data: np.ndarray):
        normalizer = {}
        data = data.flatten()
        if self.normalize == 'zscore':
            normalizer['mean'] = np.nanmean(data)
            normalizer['std'] = np.nanstd(data)
        elif self.normalize == 'robust_iqr':
            normalizer['median'] = np.median(data)
            normalizer['iqr'] = np.subtract(*np.percentile(data, [75, 25]))
        elif self.normalize == 'robust_mad':
            normalizer['median'] = np.median(data)
            normalizer['mad'] = np.median(np.abs(data - normalizer['median']))
        elif self.normalize == 'minmax':
            normalizer['min'] = np.nanmin(data)
            normalizer['max'] = np.nanmax(data)
        elif self.normalize == 'pit' or self.normalize == 'centered_pit':
            ecdf = ECDF(data)
            normalizer['ecdf'] = ecdf
        return normalizer
    
    def transform(self, data: np.ndarray, normalizer=None, data_name=None):
        # if data_name is specified, the normalizer argument will be ignored.
        assert normalizer is not None or data_name is not None, "Must specify either normalizer or data name."
        if data_name is not None:
            assert data_name in self.normalizer_dict.keys(), f"Data name({data_name}) must be in normalizer dict key."
            normalizer = self.normalizer_dict[data_name]
        if self.normalize == 'zscore':
            return (data - normalizer['mean']) / (normalizer['std'] + 1e-8)
        elif self.normalize == 'robust_iqr':
            return (data - normalizer['median']) / (normalizer['iqr'] + 1e-8)
        elif self.normalize == 'robust_mad':
            return (data - normalizer['median']) / (normalizer['mad'] + 1e-8)
        if self.normalize == 'minmax':
            return (data - normalizer['min']) / (normalizer['max'] - normalizer['min'] + 1e-8)
        elif self.normalize == 'pit' or self.normalize == 'centered_pit':
            data_shape = data.shape
            norm_data = normalizer['ecdf'](data.flatten()).reshape(data_shape)
            if self.normalize == 'centered_pit':
                norm_data = norm_data * 2 - 1
            return norm_data
        
    def inverse_transform(self, data: np.ndarray, normalizer=None, data_name=None):
        # if data_name is specified, the normalizer argument will be ignored.
        assert normalizer is not None or data_name is not None, "Must specify either normalizer or data name."
        if data_name is not None:
            assert data_name in self.normalizer_dict.keys(), f"Data name({data_name}) must be in normalizer dict key."
            normalizer = self.normalizer_dict[data_name]
        if self.normalize == 'zscore':
            return data * normalizer['std'] + normalizer['mean']
        elif self.normalize == 'robust_iqr':
            return data * normalizer['iqr'] + normalizer['median']
        elif self.normalize == 'robust_mad':
            return data * normalizer['mad'] + normalizer['median']
        if self.normalize == 'minmax':
            return data * (normalizer['max'] - normalizer['min']) + normalizer['min']
        elif self.normalize == 'pit' or self.normalize == 'centered_pit':
            ecdf: ECDF = normalizer['ecdf']
            ecdf.x[0] = ecdf.x[1]
            if self.normalize == 'centered_pit':
                data = (data + 1) / 2
            return np.interp(data, ecdf.y, ecdf.x)

        
class TSGtextDataset(Dataset):  # For generation task. Unified Univariate Generation Dataset 
    def __init__(self, data_dict: dict, text_data_dict: dict = None):
        for key, data in data_dict.items():
            assert data.ndim == 3, f"Data must be 3D, but {key} got {data.ndim}D."
        self.data_dict = data_dict
        self.text_data_dict = text_data_dict if text_data_dict else {k: np.zeros((1, 4096)) for k in data_dict.keys()}
        self.cal_data_stats()
        
    def cal_data_stats(self):
        total_items = 0
        n_items_dict = {}
        key_list = []
        key_idx_list = []
        
        for key, data in self.data_dict.items():
            num_items = data.shape[0]
            total_items += num_items
            n_items_dict[key] = num_items
            key_list.append(key)
            key_idx_list.append(total_items)
            
        self.total_items = total_items
        self.items_dict = n_items_dict    
        self.key_list = key_list
        self.key_idx_list = np.array(key_idx_list)
    
    def get_reweight_sampler(self):
        dataset_weights = np.array([1 / max(self.items_dict[key], 1) for key in self.key_list], dtype=np.float32)
        sample_weights = np.repeat(dataset_weights, [self.items_dict[key] for key in self.key_list])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=self.total_items, replacement=True)
        return sampler
        
    def __len__(self):
        return self.total_items
    
    def __getitem__(self, idx):
        assert idx < self.total_items, f"Index({idx}) must be less than number of items({self.total_items})."

        data_key = np.where(self.key_idx_list > idx)[0].min()
        data_start_idx = self.key_idx_list[data_key-1] if data_key > 0 else 0
        data: np.ndarray = self.data_dict[self.key_list[data_key]]
        text_data: np.ndarray = self.text_data_dict[self.key_list[data_key]]

        valid_idx = idx - data_start_idx
        context = data[valid_idx, :, 0]  # shape: (window,)

        if text_data.shape[0] == 0:
            control_text = np.zeros((4096,))  
        else:
            valid_idx = min(valid_idx, text_data.shape[0] - 1)
            control_text = text_data[valid_idx]

        return {
            'context': context,  # shape: (window,)
            'data_key': data_key,
            'text_embedding': control_text
        }

    
class TSGtextDataModule(pl.LightningDataModule):
    def __init__(self, data_path_dict, window=96, val_portion=0.1, as_tensor=True, normalize="centered_pit", batch_size=128, num_workers=0, pin_memory=True, drop_last=False, reweight=False, **kwargs):
        super().__init__()
        self.data_path_dict = data_path_dict
        self.data_dict = {}
        self.norm_data_dict = {}
        self.normalizer_dict = {}
        self.norm_train_dict = {}
        self.norm_val_dict = {}
        self.norm_text_data_dict = {}
        self.norm_text_train_dict = {}
        self.norm_text_val_dict = {}
        self.window = window
        self.val_portion = val_portion
        self.as_tensor = as_tensor
        assert normalize in [None, 'zscore', 'robust_iqr', 'robust_mad', 'pit', 'centered_pit', 'minmax'], f"Normalize({normalize}) must be in (zscore, robust_iqr, robust_mad, pit)."
        self.normalize = normalize
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.reweight = reweight
        self.kwargs = kwargs
        self.key_list = []
        self.drop_last = drop_last
        
    def prepare_data(self) -> None:
        print(f"Normalizing data with: {self.normalize}")
        self.key_list = []
        for data_name, data_path in self.data_path_dict.items():
            self.key_list.append(data_name)
            this_data, this_text_data = load_data_from_file(data_path)
            if this_data.ndim == 3:
                this_data = this_data.reshape(-1, this_data.shape[1], 1)
            normalizer = self.fit_normalizer(this_data)
            self.data_dict[data_name] = this_data
            self.normalizer_dict[data_name] = normalizer
            norm_data = self.transform(this_data, normalizer)
            self.norm_data_dict[data_name] = norm_data
            self.norm_text_data_dict[data_name] = this_text_data
            this_text_data = np.array(this_text_data) if isinstance(this_text_data, list) else this_text_data
            train_data, val_data = self.split_train_val(norm_data)
            train_text_data, val_text_data = self.split_train_val(this_text_data)  
            self.norm_train_dict[data_name] = train_data
            self.norm_val_dict[data_name] = val_data
            self.norm_text_train_dict[data_name] = train_text_data 
            self.norm_text_val_dict[data_name] = val_text_data 
            print(f"Loaded data: {data_name}; Train shape: {train_data.shape}, Validation shape: {val_data.shape}.")
    
    def split_train_val(self, data: np.ndarray):
        # By default, data are sliced into non-overlapped sequences.
        # shuffle stack_data, only along the first dimension
        np.random.shuffle(data)
        total_instances = data.shape[0]
        num_val_instances = int(total_instances * self.val_portion)
        train_data = data[:-num_val_instances]
        val_data = data[-num_val_instances:]
        
        return train_data, val_data
    
    def train_dataloader(self):
        train_dataset = TSGtextDataset(self.norm_train_dict, self.norm_text_train_dict)
        sampler = train_dataset.get_reweight_sampler() if self.reweight else None
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last, sampler=sampler, shuffle=not self.reweight, **self.kwargs)
    
    def val_dataloader(self):
        val_dataset = TSGtextDataset(self.norm_val_dict, self.norm_text_val_dict)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, **self.kwargs)
    
    def fit_normalizer(self, data: np.ndarray):
        normalizer = {}
        data = data.flatten()
        if self.normalize == 'zscore':
            normalizer['mean'] = np.nanmean(data)
            normalizer['std'] = np.nanstd(data)
        elif self.normalize == 'robust_iqr':
            normalizer['median'] = np.median(data)
            normalizer['iqr'] = np.subtract(*np.percentile(data, [75, 25]))
        elif self.normalize == 'robust_mad':
            normalizer['median'] = np.median(data)
            normalizer['mad'] = np.median(np.abs(data - normalizer['median']))
        elif self.normalize == 'minmax':
            normalizer['min'] = np.nanmin(data)
            normalizer['max'] = np.nanmax(data)
        elif self.normalize in ['pit', 'centered_pit']:
            normalizer['ecdf'] = ECDF(data)
        return normalizer
    
    def transform(self, data: np.ndarray, normalizer=None, data_name=None):
        # if data_name is specified, the normalizer argument will be ignored.
        assert normalizer is not None or data_name is not None, "Must specify either normalizer or data name."
        if data_name is not None:
            assert data_name in self.normalizer_dict.keys(), f"Data name({data_name}) must be in normalizer dict key."
            normalizer = self.normalizer_dict[data_name]
        if self.normalize == 'zscore':
            return (data - normalizer['mean']) / (normalizer['std'] + 1e-8)
        elif self.normalize == 'robust_iqr':
            return (data - normalizer['median']) / (normalizer['iqr'] + 1e-8)
        elif self.normalize == 'robust_mad':
            return (data - normalizer['median']) / (normalizer['mad'] + 1e-8)
        if self.normalize == 'minmax':
            return (data - normalizer['min']) / (normalizer['max'] - normalizer['min'] + 1e-8)
        elif self.normalize == 'pit' or self.normalize == 'centered_pit':
            data_shape = data.shape
            norm_data = normalizer['ecdf'](data.flatten()).reshape(data_shape)
            if self.normalize == 'centered_pit':
                norm_data = norm_data * 2 - 1
            return norm_data
        
    def inverse_transform(self, data: np.ndarray, normalizer=None, data_name=None):
        # if data_name is specified, the normalizer argument will be ignored.
        assert normalizer is not None or data_name is not None, "Must specify either normalizer or data name."
        if data_name is not None:
            assert data_name in self.normalizer_dict.keys(), f"Data name({data_name}) must be in normalizer dict key."
            normalizer = self.normalizer_dict[data_name]
        if self.normalize == 'zscore':
            return data * normalizer['std'] + normalizer['mean']
        elif self.normalize == 'robust_iqr':
            return data * normalizer['iqr'] + normalizer['median']
        elif self.normalize == 'robust_mad':
            return data * normalizer['mad'] + normalizer['median']
        if self.normalize == 'minmax':
            return data * (normalizer['max'] - normalizer['min']) + normalizer['min']
        elif self.normalize == 'pit' or self.normalize == 'centered_pit':
            ecdf: ECDF = normalizer['ecdf']
            ecdf.x[0] = ecdf.x[1]
            if self.normalize == 'centered_pit':
                data = (data + 1) / 2
            return np.interp(data, ecdf.y, ecdf.x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Linear layer

    def forward(self, x):
        return self.fc(x)

def preprocess_text_data(data):
    """Ensure all text data is valid and convert to a list of strings."""
    preprocessed_data = []
    for row in data:
        if isinstance(row, str):  # Valid string
            preprocessed_data.append(row)
        elif isinstance(row, list):  # If it's a list, join items into a string
            preprocessed_data.append(" ".join(map(str, row)))
        else:  # Convert other types to string or replace with an empty string
            preprocessed_data.append(str(row) if row is not None else "")
    return preprocessed_data

def load_data_text_from_file(file_path: str, this_window=168, text_weight=0.1, text_overall=False, text_ind=True):
    #only for testing
    print(f"this_window: {this_window}")
    
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)

        text_cols = [col for col in df.columns if '_text_description' in col]
        dataset_description_cols = [col for col in df.columns if '_dataset_description' in col]
        num_cols = [col for col in df.columns if col not in text_cols and col not in dataset_description_cols]

        # Read numeric data
        try:
            if len(num_cols) == 1:
                num_data = df[num_cols[0]].values.astype(np.float32).reshape(-1, 1)
            else:
                num_data = df[num_cols].values.astype(np.float32)
        except ValueError as e:
            print("Error converting numeric columns to float:", e)
            print("Problematic columns:", [col for col in num_cols if not pd.api.types.is_numeric_dtype(df[col])])
        
        # Load tokenizer and model
        model_name = "meta-llama/Llama-3.1-8B-Instruct" 
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        if not token:
            raise EnvironmentError("Hugging Face token not found in environment variables.")
    
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)  
        model = AutoModel.from_pretrained(model_name, use_auth_token=token)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        if text_overall:
            print("***Now is text overall***")

            # Ensure valid range for dataset description and text description columns
            for i in range(len(text_cols)):  # Iterate over text columns to avoid mismatches
                dataset_description_col = f"{i}_dataset_description"
                text_description_col = f"{i}_text_description"

                # Check if both columns exist in the dataframe
                if dataset_description_col in df.columns and text_description_col in df.columns:
                    df[text_description_col] = df[dataset_description_col] + " " + df[text_description_col]
                else:
                    print(f"Skipping index {i}: Missing columns {dataset_description_col} or {text_description_col}")
            
            # Read text data
            text_data = df[text_cols].values

            # Ensure the number of rows is a multiple of this_window
            total_rows = len(num_data)
            truncated_rows = (total_rows // this_window) * this_window
            num_data = num_data[:truncated_rows].reshape(-1, this_window, num_data.shape[1])
            text_data = text_data[:truncated_rows]

            text_data = preprocess_text_data(text_data)
            text_embeddings = []

            for i in range(0, len(text_data), this_window):
                window_text = text_data[i:i + this_window]
                first_text_in_window = window_text[0][0]
                
                inputs = tokenizer(first_text_in_window, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    word_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
                
                text_embeddings.append(word_embedding) 

            text_embeddings = np.array(text_embeddings)
            
        elif text_ind:
            print("***Now is text ind***")
            # Read text data
            text_data = df[text_cols].values
            dataset_text_data = df[dataset_description_cols].values

            # Ensure the number of rows is a multiple of this_window
            total_rows = len(num_data)
            truncated_rows = (total_rows // this_window) * this_window
            num_data = num_data[:truncated_rows].reshape(-1, this_window, num_data.shape[1])
            text_data = text_data[:truncated_rows]
            dataset_text_data = dataset_text_data[:truncated_rows]

            text_data = preprocess_text_data(text_data)
            dataset_text_data = preprocess_text_data(dataset_text_data)

            # Get text embeddings
            text_embeddings = []
            dataset_text_embeddings = []

            for i in range(0, len(text_data), this_window):
                window_text = text_data[i:i + this_window]
                first_text_in_window = window_text[0][0]

                window_dataset_text = dataset_text_data[i:i + this_window]
                first_dataset_text_in_window = window_dataset_text[0][0]
                
                inputs = tokenizer(first_text_in_window, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    word_embedding = hidden_states.mean(dim=1).squeeze().cpu().numpy()
                
                dataset_inputs = tokenizer(first_dataset_text_in_window, return_tensors="pt").to(device)
                with torch.no_grad():
                    dataset_outputs = model(**dataset_inputs)
                    dataset_hidden_states = dataset_outputs.last_hidden_state
                    dataset_word_embedding = dataset_hidden_states.mean(dim=1).squeeze().cpu().numpy()
                
                text_embeddings.append(word_embedding)
                dataset_text_embeddings.append(dataset_word_embedding)
            
            text_embeddings = np.array(text_embeddings)
            dataset_text_embeddings = np.array(dataset_text_embeddings)

        print(num_data.shape)

        return num_data, text_embeddings

    
    elif file_path.endswith(".tsf"):
        loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(
            file_path, 
            replace_missing_vals_with="NaN",
            value_column_name="series_value"
            )
        data = np.stack(loaded_data['series_value'].values).T

        num_rows = len(data)
        truncated_rows = (num_rows // this_window) * this_window
        data = data[:truncated_rows]
        
        # Reshape data into windows
        num_sequences = len(data) // this_window
        data = data.reshape(-1, this_window, 1)  
        
        # Create dummy text embeddings of the expected size
        text_embeddings = np.zeros((num_sequences, 4096))
        return data, text_embeddings  # no date column

def load_data_from_file(file_path: str):
    if file_path.endswith(".csv"):
        loaded_data = pd.read_csv(file_path)
        return loaded_data.values  # no index columns, by default.
    elif file_path.endswith(".tsf"):
        loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(
            file_path, 
            replace_missing_vals_with="NaN",
            value_column_name="series_value"
            )
        data = np.stack(loaded_data['series_value'].values).T
        return data  # no date column
    elif file_path.endswith(".npy"):
        loaded_data = np.load(file_path)  # shape like (N, T) by default
        return loaded_data.T  
    elif file_path.endswith(".pkl"):
        loaded_data = pd.read_pickle(file_path)
        return loaded_data


# Codes below are from: https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py
def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )
