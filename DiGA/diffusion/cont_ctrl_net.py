"""
Adapted from https://github.com/lucidrains/denoising-diffusion-pytorch
"""
import math
from functools import partial
from collections import namedtuple
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from torch.utils.data import Dataset, DataLoader
from statsmodels.distributions.empirical_distribution import ECDF
import pytorch_lightning as pl


from utils.metrics_utils import get_metrics_func
# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# classifier free guidance functions

def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# small helper modules

class DatasetCC(Dataset):
    def __init__(self, seq: torch.Tensor, ctrl: torch.Tensor) -> None:
        super().__init__()
        assert seq.shape[0] == ctrl.shape[0], "Sequence and class tensor must have the same length."
        self.seq = seq.clone().float()
        self.ctrl = ctrl.clone().float()

    def __len__(self) -> int:
        return self.seq.shape[0]

    def __getitem__(self, index: int) -> tuple:
        return self.seq[index], self.ctrl[index]

class DataModuleCC(pl.LightningDataModule):
    def __init__(self, train_dir, vali_dir, selected_metrics, n_bins=5, batch_size=128, normalization='zscore', cont_norm='zscore', num_workers=0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.train_data = np.load(train_dir).astype(np.float32)
        self.vali_data = np.load(vali_dir).astype(np.float32)
        self.selected_metrics = [get_metrics_func(selected_metrics)]
        self.nbins = n_bins
        self.batch_size = batch_size
        self.normalization = normalization
        self.cont_norm = cont_norm
        self.num_workers = num_workers
        self.prepare_data()
        self.kwargs = kwargs

    def prepare_data(self):
        price_changes = self.train_data[:,0,:]
        day_metrics = np.stack([metrics_func(price_changes) for metrics_func in self.selected_metrics], axis=1)
        nan_mask = np.isnan(day_metrics).any(axis=1)
        self.train_data = self.train_data[~nan_mask]
        day_metrics = day_metrics[~nan_mask]  # control targets
        self.day_metrics = day_metrics
        self.fit_scaler()
        n_bins = self.nbins
        percentiles = np.linspace(0,100,n_bins+1)
        bins_edge = np.percentile(day_metrics, percentiles)
        assert np.isnan(bins_edge).sum() == 0, "There shouldn't be any nan in the bins_edge."
        bins_edge[0], bins_edge[-1] = -np.inf, np.inf
        self.bins_edge = bins_edge
        self.bin_index = np.digitize(day_metrics, bins_edge) - 1
        self.bin_medians = [np.median(day_metrics[self.bin_index==i]) for i in range(n_bins)]
        print(f"Bin medians: {self.bin_medians}")

        vali_metrics = np.stack([metrics_func(self.vali_data[:,0,:]) for metrics_func in self.selected_metrics], axis=1)
        self.vali_metrics = vali_metrics
        self.transform()

    def fit_scaler(self):
        if self.normalization == 'zscore':
            mean, std = [], []
            for i in range(self.train_data.shape[1]):
                mean.append(np.mean(self.train_data[:,i,:]))
                std.append(np.std(self.train_data[:,i,:]))
            self.normalizer = {
                'mean': np.array(mean),
                'std': np.array(std)
            }
        elif self.normalization == 'pit':
            channel_scaler = []
            for i in range(self.train_data.shape[1]):
                c_ecdf = ECDF(self.train_data[:,i,:].flatten())
                channel_scaler.append(c_ecdf)
            self.normalizer = {
                'ecdfs': channel_scaler,
            }

        if self.cont_norm == 'zscore':
            self.cont_normalizer = {
                'mean': self.day_metrics.mean(),
                'std': self.day_metrics.std()
            }
        elif self.cont_norm == 'pit':
            self.cont_normalizer = {
                'ecdf': ECDF(self.day_metrics.flatten())
            }

    def transform(self):
        if self.normalization == 'zscore':
            self.train_data = (self.train_data - self.normalizer['mean'].reshape(1,-1,1)) / (self.normalizer['std'].reshape(1,-1,1) + 1e-6)
            self.vali_data = (self.vali_data - self.normalizer['mean'].reshape(1,-1,1)) / (self.normalizer['std'].reshape(1,-1,1) + 1e-6)
        elif self.normalization == 'pit':
            for i in range(self.train_data.shape[1]):
                self.train_data[:,i,:] = self.normalizer['ecdfs'][i](self.train_data[:,i,:])
                self.vali_data[:,i,:] = self.normalizer['ecdfs'][i](self.vali_data[:,i,:])

        if self.cont_norm == 'zscore':
            self.norm_metrics = (self.day_metrics - self.cont_normalizer['mean']) / (self.cont_normalizer['std'] + 1e-6)
            self.norm_vali_metrics = (self.vali_metrics - self.cont_normalizer['mean']) / (self.cont_normalizer['std'] + 1e-6)
            self.norm_metrics_medians = (self.bin_medians - self.cont_normalizer['mean']) / (self.cont_normalizer['std'] + 1e-6)
        elif self.cont_norm == 'pit':
            self.norm_metrics = self.cont_normalizer['ecdf'](self.day_metrics)
            self.norm_vali_metrics = self.cont_normalizer['ecdf'](self.vali_metrics)
            self.norm_metrics_medians = self.cont_normalizer['ecdf'](self.bin_medians)
        print(f"Norm bin medians: {self.norm_metrics_medians}")

    def inverse_transform(self, data):
        if self.normalization == 'zscore':
            data = data * self.normalizer['std'].reshape(1,-1,1) + self.normalizer['mean'].reshape(1,-1,1)
        elif self.normalization == 'pit':
            for i in range(data.shape[1]):
                data[:,i,:] = self.normalizer['ecdfs'][i](data[:,i,:])
        return data

    def transform_cont(self, data):
        if self.cont_norm == 'zscore':
            data = (data - self.cont_normalizer['mean']) / (self.cont_normalizer['std'] + 1e-6)
        elif self.cont_norm == 'pit':
            data = self.cont_normalizer['ecdf'](data)
        return data

    def train_dataloader(self):
        train_dataset = DatasetCC(torch.from_numpy(self.train_data), torch.from_numpy(self.norm_metrics))
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True, drop_last=True, **self.kwargs)

    def val_dataloader(self):
        vali_dataset = DatasetCC(torch.from_numpy(self.vali_data), torch.from_numpy(self.norm_vali_metrics))
        return DataLoader(vali_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, **self.kwargs)

    def test_dataloader(self):  # not used
        vali_dataset = DatasetCC(torch.from_numpy(self.vali_data), torch.from_numpy(self.norm_vali_metrics))
        return DataLoader(vali_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=False, **self.kwargs)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None, norm = None):
    if norm == 'bn':
        norm_func = nn.BatchNorm1d(default(dim_out, dim))
    elif norm == 'ln':
        norm_func = LayerNorm(default(dim_out, dim))
    else:
        norm_func = nn.Identity()
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, default(dim_out, dim), 15, padding = 7),
        norm_func,
        nn.SiLU()
    )

def Downsample(dim, dim_out = None, norm = None):
    if norm == 'bn':
        norm_func = nn.BatchNorm1d(default(dim_out, dim))
    elif norm == 'ln':
        norm_func = LayerNorm(default(dim_out, dim))
    else:
        norm_func = nn.Identity()
    return nn.Sequential(
        nn.Conv1d(dim, default(dim_out, dim), 30, 2, 14),
        norm_func,
        nn.SiLU()
    )

class LastConv(nn.Module):
    def __init__(self, dim, dim_out, norm = None):
        super().__init__()
        if norm == 'bn':
            self.norm_func = nn.BatchNorm1d(dim_out)
        elif norm == 'ln':
            self.norm_func = LayerNorm(dim_out)
        else:
            self.norm_func = nn.Identity()

        self.conv = nn.Conv1d(dim, dim_out, 15, padding = 7)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm_func(self.conv(x)))

class WeightStandardizedConv2d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 15, padding = 7)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None, groups = 8, norm = 'bn'):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(dim_out)
        elif norm == 'ln':
            self.norm = LayerNorm(dim_out)
        else:
            self.norm = nn.Identity()
        self.act = nn.SiLU()
    def forward(self, x, time_emb = None, class_emb = None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 ')
            scale_shift = cond_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return self.act(self.norm(h + self.res_conv(x)))

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.act(self.to_out(out))

class CtrlEmbedding(nn.Module):
    def __init__(self, n_ctrls, dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.embed = nn.Sequential(
            nn.Linear(n_ctrls, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        embed_out = self.embed(x)
        return embed_out
# model
class LearnedSinusoidalCtrlEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x:torch.Tensor):
        if x.dim == 1:
            x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered.squeeze()

class SinusoidalCtrlEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.sin_emb = SinusoidalPosEmb(dim)

    def forward(self, x):
        emb = self.sin_emb(x)
        return emb.squeeze()

class UnetCC(nn.Module):
    def __init__(
        self,
        dim,
        n_ctrls=1,
        cond_drop_prob = 0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        inter_normalization = 'ln',
        cont_emb = 'l'
    ):
        super().__init__()

        # classifier free guidance stuff

        self.cond_drop_prob = cond_drop_prob

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 15, padding = 7)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups, norm = inter_normalization)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # class embeddings
        cont_emb_dim = dim
        if cont_emb == 'l':
            self.classes_emb = CtrlEmbedding(n_ctrls, dim)
        elif cont_emb == 's':
            self.classes_emb = SinusoidalCtrlEmb(dim)
        elif cont_emb == 'ls':
            self.classes_emb = LearnedSinusoidalCtrlEmb(dim)
            cont_emb_dim = cont_emb_dim + 1
        self.null_classes_emb = nn.Parameter(torch.randn(cont_emb_dim))

        classes_dim = dim * 4

        self.classes_mlp = nn.Sequential(
            nn.Linear(cont_emb_dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out, inter_normalization) if not is_last else LastConv(dim_in, dim_out, inter_normalization)  # nn.Conv1d(dim_in, dim_out, 15, padding = 7)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in, inter_normalization) if not is_last else LastConv(dim_in, dim_out, inter_normalization)  # nn.Conv1d(dim_out, dim_in, 15, padding = 7)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        rescaled_phi = 0.,
        **kwargs
    ):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def forward(self, x, time, classes, cond_drop_prob = None, x_self_cond = None):
        batch, device = x.shape[0], x.device

        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # derive condition, with condition dropout for classifier free guidance
        classes_emb = self.classes_emb(classes)

        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, c)
        return self.final_conv(x)
