from utils.metrics_utils import *
import numpy as np
from scipy.stats import wasserstein_distance, entropy, ks_2samp
from scipy.spatial.distance import jensenshannon
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def cal_distances(gt, sp):
    gt = gt[~np.isnan(gt)]
    sp = sp[~np.isnan(sp)]
    w1d = wasserstein_distance(gt, sp)
    ks = ks_2samp(gt, sp)
    hist_real, edge_real = np.histogram(gt, density=True, bins=50)
    hist_gen, _ = np.histogram(sp, density=True, bins=edge_real)
    kl = entropy(hist_real, hist_gen+1e-9)
    js = jensenshannon(hist_real, hist_gen+1e-9)
    return {'w1d': w1d,
            'ks': ks.statistic,
            'kl': kl,
            'js': js}

def get_test_funcs(data_path):
    train_data = np.load(data_path)
    price_changes = train_data[:,0,:]
    sample_gt_chg = price_changes.flatten()

    gt_day_return = batch_return(price_changes)
    gt_day_volatility = batch_volatility(price_changes)
    gt_day_amplitude = batch_amplitude(price_changes)
    gt_day_kurtosis = batch_kurtosis(price_changes)
    gt_day_skew = batch_skew(price_changes)
    gt_day_sel_acf = batch_selected_acf(price_changes)
    gt_day_sel_clust = batch_selected_vol_clust(price_changes)


    def plot_density_compare(gt, sp, minq=0, maxq=1, title='', bins=50):
        bin_edges = np.linspace(np.nanquantile(gt, minq), np.nanquantile(gt, maxq), bins)
        sns.histplot(sp, bins=bin_edges, color='red', label='samples', stat='probability', alpha=0.6)
        sns.histplot(gt, bins=bin_edges, color='blue', label='ground truth', stat='probability', alpha=0.6)
        plt.title(title)
        plt.legend()
        plt.show()


    def plot_density_compare_on_ax(gt, sp, ax:plt.Axes, minq=0, maxq=1, title='', bins=50, kde=True):
        bin_edges = np.linspace(np.nanquantile(gt, minq), np.nanquantile(gt, maxq), bins)
        sns.histplot(sp, bins=bin_edges, color='red', label='samples', stat='density', alpha=0.6, ax=ax, kde=kde, kde_kws={'clip':(bin_edges.min(), bin_edges.max())})
        sns.histplot(gt, bins=bin_edges, color='blue', label='ground truth', stat='density', alpha=0.6, ax=ax, kde=kde, kde_kws={'clip':(bin_edges.min(), bin_edges.max())})
        ax.set_title(title)
        ax.legend()

    def plot_compare_gt_sp(sample_price_changes):
        # plot metrics distributions of given samples from the same model checkpoint and the ground truth
        # at the same time, calculate the distance metrics between the two distributions
        sp_day_return = batch_return(sample_price_changes)
        sp_day_volatility = batch_volatility(sample_price_changes)
        sp_day_amplitude = batch_amplitude(sample_price_changes)
        sp_day_kurtosis = batch_kurtosis(sample_price_changes)
        sp_day_skew = batch_skew(sample_price_changes)
        sp_day_sel_acf = batch_selected_acf(sample_price_changes)
        sp_day_sel_clust = batch_selected_vol_clust(sample_price_changes)

        dist_dict = {}
        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        plot_density_compare_on_ax(gt_day_return, sp_day_return, ax=axes[0,0], minq=0.01, maxq=0.99, title='Daily Return')
        dist_dict['return'] = cal_distances(gt_day_return, sp_day_return)
        plot_density_compare_on_ax(gt_day_volatility, sp_day_volatility, ax=axes[0,1], minq=0.01, maxq=0.99, title='Daily Volatility')
        dist_dict['volatility'] = cal_distances(gt_day_volatility, sp_day_volatility)
        plot_density_compare_on_ax(gt_day_amplitude, sp_day_amplitude, ax=axes[0,2], minq=0.01, maxq=0.99, title='Daily Amplitude')
        dist_dict['amplitude'] = cal_distances(gt_day_amplitude, sp_day_amplitude)
        plot_density_compare_on_ax(gt_day_kurtosis, sp_day_kurtosis, ax=axes[0,3], minq=0.01, maxq=0.99, title='Daily Kurtosis')
        dist_dict['kurtosis'] = cal_distances(gt_day_kurtosis, sp_day_kurtosis)
        plot_density_compare_on_ax(gt_day_skew, sp_day_skew, ax=axes[0,4], minq=0.01, maxq=0.99, title='Daily Skewness')
        dist_dict['skewness'] = cal_distances(gt_day_skew, sp_day_skew)

        selections = [1,5,15,30,60]
        for sel in range(5):
            plot_density_compare_on_ax(gt_day_sel_acf[sel], sp_day_sel_acf[sel], ax=axes[1,sel], minq=0.01, maxq=0.99, title=f'Daily ACF with lag {selections[sel]}')
            dist_dict[f'acf_lag{selections[sel]}'] = cal_distances(gt_day_sel_acf[sel], sp_day_sel_acf[sel])

        selections = [1,5,15,30,60]
        for sel in range(5):
            plot_density_compare_on_ax(gt_day_sel_clust[sel], sp_day_sel_clust[sel], ax=axes[2,sel], minq=0.01, maxq=0.99, title=f'Daily volatility clustering with lag {selections[sel]}')
            dist_dict[f'vol_clust_lag{selections[sel]}'] = cal_distances(gt_day_sel_clust[sel], sp_day_sel_clust[sel])

        plt.tight_layout()
        return fig, dist_dict

    def plot_compare_mid(flat_gen_chg, ax:Optional[plt.Axes]=None):
        if ax is None:
            fig = plt.figure(figsize=(6,4))
            ax = plt.gca()
        else :
            fig = ax.get_figure()
        plot_density_compare_on_ax(sample_gt_chg, flat_gen_chg, ax=ax, minq=0.02, maxq=0.98, title='Mid Price',bins=50)
        return fig, cal_distances(sample_gt_chg, flat_gen_chg)

    return plot_compare_gt_sp, plot_compare_mid, plot_density_compare, plot_density_compare_on_ax


def get_all_bin_medians(data_path=Path('/data/container/amlt_data'), n_bins=5):
    median_dict = {}
    for dataset in ['SZAMain', 'ChiNext']:
        median_dict[dataset] = {}
        dataset_path = data_path / f'{dataset}_train.npy'
        train_data = np.load(dataset_path)
        price_changes = train_data[:,0,:]
        for ctrl_target in ['return', 'amplitude', 'volatility', 'kurtosis']:
            median_dict[dataset][ctrl_target] = {}
            selected_metrics = get_metrics_func(ctrl_target)
            day_metrics = selected_metrics(price_changes)
            nan_mask = np.isnan(day_metrics)

            price_changes = price_changes[~nan_mask]
            day_metrics = day_metrics[~nan_mask]
            mean_metrics, std_metrics = day_metrics.mean(), day_metrics.std()

            percentiles = np.linspace(0,100,n_bins+1)
            bins_edge = np.percentile(day_metrics, percentiles)
            bins_edge[0], bins_edge[-1] = -np.inf, np.inf
            bin_index = np.digitize(day_metrics, bins_edge) - 1
            bin_medians = [np.median(day_metrics[bin_index==i]) for i in range(n_bins)]
            median_dict[dataset][ctrl_target]['medians'] = bin_medians
            median_dict[dataset][ctrl_target]['bins_edge'] = bins_edge
            median_dict[dataset][ctrl_target]['bin_index'] = bin_index
            median_dict[dataset][ctrl_target]['metrics'] = day_metrics
            median_dict[dataset][ctrl_target]['mean_std'] = mean_metrics, std_metrics
    return median_dict
