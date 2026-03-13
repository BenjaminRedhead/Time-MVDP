# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from functools import partial
from typing import Callable, Optional
import os

import hydra
import lightning as L
import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils._pytree import tree_map
from torch.utils.data import Dataset, DistributedSampler

from tsfm.common import hydra_util  # noqa: hydra resolvers
from tsfm.data.loader import DataLoader


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset | list[Dataset]],
        data_builder=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.data_builder = data_builder

        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader

    @staticmethod
    def get_dataloader(
        dataset: Dataset,
        dataloader_func: Callable[..., DataLoader],
        shuffle: bool,
        world_size: int,
        batch_size: int,
        num_batches_per_epoch: Optional[int] = None,
    ) -> DataLoader:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=None,
                rank=None,
                shuffle=shuffle,
                seed=0,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        return dataloader_func(
            dataset=dataset,
            shuffle=shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            self.train_dataset,
            instantiate(self.cfg.train_dataloader, _partial_=True),
            self.cfg.train_dataloader.shuffle,
            self.trainer.world_size,
            self.train_batch_size,
            num_batches_per_epoch=self.train_num_batches_per_epoch,
        )

    @staticmethod
    def get_torch_dataloader(
        dataset: Dataset,
        dataloader_func: Callable[..., DataLoader],
        shuffle: bool,
        world_size: int,
        batch_size: int,
        num_batches_per_epoch: Optional[int] = None,
    ) -> DataLoader:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=None,
                rank=None,
                shuffle=shuffle,
                seed=0,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        return dataloader_func(
            dataset=dataset,
            shuffle=shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=batch_size,
        )
    
    def _val_dataloader(self) -> DataLoader | list[DataLoader]:
        return tree_map(
            partial(
                self.get_torch_dataloader,
                dataloader_func=instantiate(self.cfg.val_dataloader, _partial_=True),
                shuffle=self.cfg.val_dataloader.shuffle,
                world_size=self.trainer.world_size,
                batch_size=self.val_batch_size,
                num_batches_per_epoch=None,
            ),
            self.val_dataset,
        )

    @property
    def train_batch_size(self) -> int:
        return self.cfg.train_dataloader.batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def val_batch_size(self) -> int:
        return self.cfg.val_dataloader.batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def train_num_batches_per_epoch(self) -> int:
        return (
            self.cfg.train_dataloader.num_batches_per_epoch
            * self.trainer.accumulate_grad_batches
        )


@hydra.main(version_base="1.3", config_name="default.yaml")
def main(cfg: DictConfig):
    if cfg.tf32:
        assert cfg.trainer.precision == 32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model: L.LightningModule = instantiate(cfg.model, _convert_="all")

    if cfg.compile:
        model.module.compile(mode=cfg.compile)
    trainer: L.Trainer = instantiate(cfg.trainer)
    
    # Instantiate the data builder and create dataset
    data_builder = instantiate(cfg.data)
    train_dataset: Dataset = data_builder.load_dataset(model.train_transform_map)
    
    val_dataset: Optional[Dataset | list[Dataset]] = (
        tree_map(
            lambda ds: ds.load_dataset(model.val_transform_map),
            instantiate(cfg.val_data, _convert_="all"),
        )
        if "val_data" in cfg
        else None
    )
    L.seed_everything(cfg.seed, workers=True)
    
    print("train_dataset size:", len(train_dataset))
    
    datamodule = DataModule(cfg, train_dataset, [val_dataset[0]], data_builder)

    # only work when epoch=0
    trainer.fit(
        model,
        datamodule=datamodule,  # Pass data_builder
        ckpt_path=cfg.ckpt_path,
    )

    # Initialize dictionary to collect processed patches by dataset
    dataset_patches = {}
    
    print("Processing training dataloader...")
    print("Expected to process approximately 15000 samples")
    
    total_samples = 0
    
    # Go through the whole training dataloader
    for batch_idx, batch in enumerate(datamodule.train_dataloader()):
        if batch_idx % 100 == 0:
            print(f"Processing batch {batch_idx}, total samples so far: {total_samples}")
        
        # Get the label data - shape should be (batch_size, 512, 32)
        labels = batch['label']  # (batch_size, 512, 32)
        batch_size = labels.shape[0]
        
        # Get the time_id data to check for valid positions
        time_ids = batch['time_id']  # (batch_size, 512)
        
        # Get the dataset indices to determine which subdataset each sample belongs to
        dataset_indices = batch.get('_dataset_idx', None)
        
        # Process each sample in the batch
        for sample_idx in range(batch_size):
            label_sample = labels[sample_idx]  # (512, 32)
            time_id_sample = time_ids[sample_idx]  # (512,)
            
            # Determine which subdataset this sample belongs to
            if dataset_indices is not None:
                global_idx = dataset_indices[sample_idx].item()
                # Get dataset name using the data_builder's method
                if hasattr(data_builder, 'get_dataset_name_for_global_index'):
                    dataset_name = data_builder.get_dataset_name_for_global_index(global_idx)
                else:
                    dataset_name = f"dataset_{global_idx}"
            else:
                dataset_name = "train"
            
            # Initialize list for this dataset if not exists
            if dataset_name not in dataset_patches:
                dataset_patches[dataset_name] = []
            
            # Sample 10 successive patches from the second dimension (512)
            # We need to ensure we don't go out of bounds and time_ids are valid
            if label_sample.shape[0] >= 10:
                # Find valid starting positions where time_id is non-zero for at least 10 consecutive positions
                # (except possibly the first one can be 0)
                valid_start_positions = []
                
                for potential_start in range(label_sample.shape[0] - 9):
                    # Check if the next 9 positions (after the first) have non-zero time_id
                    time_slice = time_id_sample[potential_start:potential_start + 10]
                    # Allow first one to be 0, but the rest should be non-zero
                    if torch.sum(time_slice[1:] != 0) == 9:  # All 9 positions after first should be non-zero
                        valid_start_positions.append(potential_start)
                
                if len(valid_start_positions) > 0:
                    # Randomly select from valid positions
                    start_idx = valid_start_positions[torch.randint(0, len(valid_start_positions), (1,)).item()]
                    
                    # Extract 10 successive patches
                    patches = []
                    for i in range(10):
                        patch = label_sample[start_idx + i]  # (32,)
                        patches.append(patch)
                    
                    # Concatenate the 10 patches: [patch1(32), patch2(32), ..., patch10(32)]
                    concatenated_patches = torch.cat(patches, dim=0)  # (320,)
                    
                    # Reshape to (320, 1) and add batch dimension to get (1, 320, 1)
                    processed_sample = concatenated_patches.unsqueeze(-1).unsqueeze(0)  # (1, 320, 1)
                    
                    # Convert to numpy and add to collection for this dataset
                    dataset_patches[dataset_name].append(processed_sample.cpu().numpy())
                    total_samples += 1
            if len(dataset_patches[dataset_name]) == 0:
                print(f"Skipping sample {sample_idx} from dataset {dataset_name} because it has less than 10 valid positions")
                print("The valid start positions are: ", valid_start_positions)
    
    print(f"Total samples processed: {total_samples}")
    print(f"Found {len(dataset_patches)} subdatasets")
    
    # Save each subdataset's patches to a separate file
    for dataset_name, patches_list in dataset_patches.items():
        if patches_list:
            # Concatenate all samples along the first dimension
            final_array = np.concatenate(patches_list, axis=0)  # (total_samples, 320, 1)
            print(f"Dataset '{dataset_name}': {final_array.shape}")
            
            # Save to npy file in the analysis folder
            output_path = os.path.join(os.path.dirname(__file__), f"extracted_label_patches_{dataset_name}.npy")
            np.save(output_path, final_array)
            print(f"Saved processed patches for '{dataset_name}' to: {output_path}")
        else:
            print(f"No samples were processed successfully for dataset '{dataset_name}'")


if __name__ == "__main__":
    main() 