# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from pathlib import Path
from diffusion.utils.pkl_utils import save_pkl
from diffusion.metrics.metrics_sets import run_metrics, calculate_one
from diffusion.ldm.data.tsg_dataset import TSGDataset
import os
from diffusion.ldm.modules.guidance_scorer import GradDotCalculator
from diffusion.classifier.model import RNNClassifier
from diffusion.classifier.classifier_train import TimeSeriesDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
from collections import Counter
import pickle as pkl

data_root = os.environ['DATA_ROOT']

def test_model_with_dp(model, data, trainer, opt, logdir, use_pam=False, use_text=False):
    """
    Testing with domain prompts (dp), optionally text-conditioned.
    """
    # === Load best checkpoint === #
    if trainer.callbacks[-1].best_model_path:
        best_ckpt_path = trainer.callbacks[-1].best_model_path
        print(f"[TEST DP] Loading best model from {best_ckpt_path}")
        model.init_from_ckpt(best_ckpt_path)

    model = model.cuda().eval()

    save_path = Path(logdir) / 'generated_samples'
    save_path.mkdir(exist_ok=True, parents=True)

    seq_len = data.window
    num_dp = 100
    all_metrics = {}

    for dataset in data.norm_train_dict.keys():
        print(f"\n[Processing dataset]: {dataset}")

        dataset_data = TSGDataset(
            {dataset: data.norm_train_dict[dataset]},
            {dataset: data.norm_text_train_dict[dataset]} if use_text else None
        )

        dataset_samples = []
        text_embeddings = []

        if use_pam:
            for idx in np.random.randint(len(dataset_data), size=num_dp):
                sample = dataset_data[idx]
                dataset_samples.append(sample['context'])
                if use_text:
                    text_embeddings.append(sample['text_embedding'])

            dataset_samples = np.vstack(dataset_samples)
            dataset_samples = torch.tensor(dataset_samples).to('cuda').float().unsqueeze(1)

            if use_text:
                text_embeddings = np.vstack(text_embeddings)
                text_embeddings = torch.tensor(text_embeddings).to('cuda').float()

        else:
            dataset_samples = None
            text_embeddings = None

        # === Get Conditioning === #
        c, mask = model.get_learned_conditioning(dataset_samples, return_mask=True)

        repeats = int(1000 / (dataset_samples.shape[0] if dataset_samples is not None else 1)) if not opt.debug else 1
        cond = torch.repeat_interleave(c, repeats, dim=0)
        mask_repeat = torch.repeat_interleave(mask, repeats, dim=0) if mask is not None else None

        all_gen = []
        for _ in range(5 if not opt.debug else 1):
            samples, _ = model.sample_log(
                cond=cond,
                batch_size=cond.shape[0],
                ddim=False,
                cfg_scale=1,
                mask=mask_repeat,
                text_embedding=text_embeddings if use_text else None
            )

            norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
            inv_samples = data.inverse_transform(norm_samples, data_name=dataset)
            all_gen.append(inv_samples)

        generated_data = np.vstack(all_gen).transpose(0, 2, 1)

        tmp_name = f'{dataset}_{seq_len}_generation_with_text' if use_text else f'{dataset}_{seq_len}_generation'
        np.save(save_path / f'{tmp_name}.npy', generated_data)

        all_metrics = run_metrics(
            data_name=dataset,
            seq_len=seq_len,
            model_name=tmp_name,
            gen_data=generated_data,
            scale='zscore',
            exist_dict=all_metrics
        )

    print(f"\n[Metrics Summary]: {all_metrics}")
    save_pkl(all_metrics, save_path / 'metric_dict.pkl')


def test_model_uncond(model, data, trainer, opt, logdir):
    if trainer.callbacks[-1].best_model_path:
        best_ckpt_path = trainer.callbacks[-1].best_model_path
        print(f"Loading best model from {best_ckpt_path} for sampling")
        model.init_from_ckpt(best_ckpt_path)

    model = model.cuda().eval()
    save_path = Path(logdir) / 'generated_samples'
    save_path.mkdir(exist_ok=True, parents=True)

    seq_len = data.window
    all_metrics = {}

    for dataset in data.norm_train_dict:
        all_gen = []
        for _ in range(5 if not opt.debug else 1):
            samples, _ = model.sample_log(
                cond=None,
                batch_size=1000 if not opt.debug else 100,
                ddim=False,
                cfg_scale=1
            )
            norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
            inv_samples = data.inverse_transform(norm_samples, data_name=dataset)
            all_gen.append(inv_samples)

        generated_data = np.vstack(all_gen).transpose(0, 2, 1)

        tmp_name = f'{dataset}_{seq_len}_uncond_generation'
        np.save(save_path / f'{tmp_name}.npy', generated_data)

        all_metrics = run_metrics(
            data_name=dataset,
            seq_len=seq_len,
            model_name=tmp_name,
            gen_data=generated_data,
            scale='zscore',
            exist_dict=all_metrics
        )

    print(f"\n[Unconditional Metrics Summary]: {all_metrics}")
    save_pkl(all_metrics, save_path / 'metric_dict.pkl')



def test_model_guidance(model, data, trainer, opt, logdir):
    if trainer.callbacks[-1].best_model_path:
        best_ckpt_path = trainer.callbacks[-1].best_model_path
        print(f"Loading best model from {best_ckpt_path} for sampling")
        model.init_from_ckpt(best_ckpt_path)

    model = model.cuda().eval()
    save_path = Path(logdir) / 'generated_samples'
    save_path.mkdir(exist_ok=True, parents=True)
    ### DEMO MODEL ###
    downstream_model = RNNClassifier(
        input_dim=7,
        hidden_dim=256,
        num_layers=2,
        rnn_type='gru',
        num_classes=1,
    )

    downstream_model.load_state_dict(
        torch.load(opt.downstream_pth_path)['model_state'])
    print('#### Downstream Model Loaded #####')
    alpha = opt.guidance_scale
    print("#### Start Generating Samples #####")
    train_tuple = pd.read_pickle(opt.guidance_path)
    generation_nums_label = dict(Counter(train_tuple[1]))

    for dataset in data.norm_train_dict:
        normalizer = data.normalizer_dict[dataset]
        train_dataset = TimeSeriesDataset(data.transform(
            train_tuple[0].transpose(0, 2, 1), normalizer),
                                          train_tuple[1],
                                          normalize=False)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=1,
                                      shuffle=False)
        c = GradDotCalculator(downstream_model, train_dataloader,
                              nn.BCEWithLogitsLoss(), alpha)
        generated_data_all = None

        print(f"#### Start Generating Samples with alpha {alpha} #####")
        for label_sample, total_samples in tqdm(generation_nums_label.items()):
            label = torch.tensor([label_sample] * total_samples,
                                 device='cuda',
                                 dtype=torch.long)
            samples, z_denoise_row = model.sample_log(cond=None,
                                                      batch_size=total_samples,
                                                      ddim=True,
                                                      ddim_steps=20,
                                                      eta=1.,
                                                      sem_guide=True,
                                                      sem_guide_type='GDC',
                                                      label=label,
                                                      GDCalculater=c)
            norm_samples = model.decode_first_stage(
                samples).detach().cpu().numpy()
            inv_samples = data.inverse_transform(norm_samples,
                                                 data_name=dataset)
            generated_data = np.array(inv_samples).transpose(0, 2, 1)
            if generated_data_all is None:
                generated_data_all = generated_data
            else:
                generated_data_all = np.concatenate(
                    [generated_data_all, generated_data], axis=0)
        generated_data_all = generated_data_all.transpose(0, 2, 1)
        label = np.concatenate([
            np.full(count, label)
            for label, count in generation_nums_label.items()
        ])
        tmp_name = f'synt_tardiff_noise_rnn_train_guidance_sc{alpha}'
        with open(save_path /
                  f'{tmp_name}.pkl', 'wb') as f:
            #with open(save_path / f'alpha_search/{tmp_name}.pkl', 'wb') as f:

            pkl.dump((generated_data_all, label), f)
        print(f"Saved {tmp_name}.pkl in {save_path}")


def zero_shot_k_repeat(samples, model, train_data_module, num_gen_samples=1000, use_text=False, text_emb=None):
    data = train_data_module
    k_samples = samples.transpose(0, 2, 1)
    k = k_samples.shape[0]

    normalizer = data.fit_normalizer(k_samples)
    norm_k_samples = data.transform(k_samples, normalizer=normalizer)

    x = torch.tensor(norm_k_samples).float().to('cuda')

    c, mask = model.get_learned_conditioning(x, return_mask=True)

    repeats = int(num_gen_samples / k)
    extra = num_gen_samples - repeats * k

    cond = torch.repeat_interleave(c, repeats, dim=0)
    cond = torch.cat([cond, c[:extra]], dim=0)

    mask_repeat = torch.repeat_interleave(mask, repeats, dim=0)
    mask_repeat = torch.cat([mask_repeat, mask[:extra]], dim=0)

    samples, _ = model.sample_log(
        cond=cond,
        batch_size=cond.shape[0],
        ddim=False,
        cfg_scale=1,
        mask=mask_repeat,
        text_embedding=text_emb if use_text else None
    )

    norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
    inv_samples = data.inverse_transform(norm_samples, normalizer=normalizer)

    gen_data = inv_samples.transpose(0, 2, 1)
    return gen_data, k_samples.transpose(0, 2, 1)


def merge_dicts(dicts):
    result = {}
    for d in dicts:
        for k, v in d.items():
            result[k] = v
    return result


def test_model_unseen(model, data, trainer, opt, logdir, use_text=False):
    """
    Zero-shot test on unseen datasets, optionally with text.
    """
    all_metrics = {}
    seq_len = opt.seq_len
    unseen_domains = ['stock', 'web']
    total_samples = 2000

    for data_name in unseen_domains:
        data_result_dicts = []

        uni_ori_data_path = f'{data_root}/ts_data/new_zero_shot_data/{data_name}_{seq_len}_test_sample.npy'
        uni_ori_data = np.load(uni_ori_data_path)

        if data_name == 'web':
            uni_ori_data = uni_ori_data[uni_ori_data < np.percentile(uni_ori_data, 99)]

        uni_data_mean = np.mean(uni_ori_data)
        uni_data_std = np.std(uni_ori_data)
        uni_scaled_ori = (uni_ori_data - uni_data_mean) / (uni_data_std + 1e-7)

        print(f"[{data_name}] Unseen test sample shape: {uni_ori_data.shape}")

        for k in [3, 10, 100]:
            k_sample_path = f'{data_root}/ts_data/new_zero_shot_data/{data_name}_{seq_len}_k_{k}_sample.npy'
            k_samples = np.load(k_sample_path)

            zero_shot_dataset = TSGDataset(
                {data_name: k_samples[:, :, np.newaxis]},
                {data_name: np.zeros((k_samples.shape[0], 4096))} if use_text else None
            )

            dataset_samples = []
            text_embeddings = []

            for idx in range(len(zero_shot_dataset)):
                sample = zero_shot_dataset[idx]
                dataset_samples.append(sample['context'])
                if use_text:
                    text_embeddings.append(sample['text_embedding'])

            dataset_samples = np.vstack(dataset_samples)
            dataset_samples = torch.tensor(dataset_samples).to('cuda').float().unsqueeze(1)

            if use_text:
                text_embeddings = np.vstack(text_embeddings)
                text_embeddings = torch.tensor(text_embeddings).to('cuda').float()

            c, mask = model.get_learned_conditioning(dataset_samples, return_mask=True)

            repeats = int(total_samples / len(zero_shot_dataset))
            cond = torch.repeat_interleave(c, repeats, dim=0)
            mask_repeat = torch.repeat_interleave(mask, repeats, dim=0)

            samples, _ = model.sample_log(
                cond=cond,
                batch_size=cond.shape[0],
                ddim=False,
                cfg_scale=1,
                mask=mask_repeat,
                text_embedding=text_embeddings if use_text else None
            )

            norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
            inv_samples = data.inverse_transform(norm_samples, data_name=data_name)

            gen_data = inv_samples.transpose(0, 2, 1)

            save_gen_path = Path(logdir) / f"generated_samples/{data_name}_{seq_len}_k{k}_repeat_gen.npy"
            np.save(save_gen_path, gen_data)

            print(f"Saved generated samples to {save_gen_path}")

            metrics = calculate_one(
                gen_data.squeeze(),
                uni_scaled_ori.squeeze(),
                '',
                0,
                f"{data_name}_{k}",
                seq_len,
                uni_data_mean,
                uni_data_std,
                total_samples
            )

            data_result_dicts.append(metrics)

        data_metrics = merge_dicts(data_result_dicts)
        all_metrics.update(data_metrics)

    print(f"\n[Unseen Domain Metrics Summary]: {all_metrics}")
    save_pkl(all_metrics, Path(logdir) / 'unseen_domain_metric_dict.pkl')
