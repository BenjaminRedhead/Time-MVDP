import numpy as np
from scipy.stats import kurtosis, skew, pearsonr
from statsmodels.tsa.stattools import acf

def single_return(min_pc_arr):
    """Caculate the return rate of a day on one price changes series.

    Input shape: (seq_len,) where seq_len is the number of observations.
    Output shape: (1, )
    """
    return np.cumsum(min_pc_arr)[-1]

def batch_return(min_pc_arr):
    """Caculate the return rate of a day on a batch of price changes series.

    Input shape: (N, seq_len) where seq_len is the number of observations.
    Output shape: (N,)
    """
    return np.cumsum(min_pc_arr, axis=1)[:,-1]

def single_volatility(min_pc_arr):
    """Caculate the return of a day on one price changes series.

    Input shape: (seq_len,) where seq_len is the number of observations.
    Output shape: (1, )
    """
    return np.std(min_pc_arr)

def batch_volatility(min_pc_arr):
    """Caculate the volatility of a day on a batch of price changes series.

    Input shape: (N, seq_len) where seq_len is the number of observations.
    Output shape: (N,)
    """
    return np.std(min_pc_arr,axis=1)

def single_amplitude(min_pc_arr):
    """Caculate the amplitude of a day on one price changes series.

    Input shape: (seq_len,) where seq_len is the number of observations.
    Output shape: (1, )
    """
    return_by_min = np.cumsum(min_pc_arr)
    return np.max(return_by_min) - np.min(return_by_min)

def batch_amplitude(min_pc_arr):
    """Caculate the amplitude of a day on a batch of price changes series.

    Input shape: (N, seq_len) where seq_len is the number of observations.
    Output shape: (N,)
    """
    return_by_min = np.cumsum(min_pc_arr, axis=1)
    return np.max(return_by_min, axis=1) - np.min(return_by_min, axis=1)

def single_kurtosis(min_pc_arr):
    """Caculate the kurtosis of a day on one price changes series.

    Input shape: (seq_len,) where seq_len is the number of observations.
    Output shape: (1, )
    """
    return kurtosis(min_pc_arr)

def batch_kurtosis(min_pc_arr):
    """Caculate the kurtosis of a day on a batch of price changes series.

    Input shape: (N, seq_len) where seq_len is the number of observations.
    Output shape: (N,)
    """
    return kurtosis(min_pc_arr, axis=1)

def single_skew(min_pc_arr):
    """Calculate the skewness of a day on one price changes series.

    Input shape: (seq_len,) where seq_len is the number of observations.
    Output shape: (1, )
    """
    return skew(min_pc_arr)

def batch_skew(min_pc_arr):
    """Caculate the skewness of a day on a batch of price changes series.

    Input shape: (N, seq_len) where seq_len is the number of observations.
    Output shape: (N,)
    """
    return skew(min_pc_arr, axis=1)

def single_acf(min_pc_arr, lags=[1,5,15,30,60]):
    """Caculate the ACF of a day on one price changes series, given ascending lags.

    Input shape: (seq_len, ) where seq_len is the number of observations.
    Output shape: list of len(lags)
    """
    all_acf = acf(min_pc_arr, nlags=lags[-1])
    return [all_acf[i] for i in lags]

def batch_acf(min_pc_arr, lag=1):
    """Caculate the ACF of a day on a batch of price changes series, given one specific lag.

    Input shape: (N, seq_len) where seq_len is the number of observations.
    Output shape: (N, 1)

    Verified to yield same results as statsmodels.tsa.stattools.acf given the same lag.
    """
    sample_mean_adj = min_pc_arr - min_pc_arr.mean(axis=-1, keepdims=True)
    sample_var = np.var(min_pc_arr, axis=-1)
    self_cov = np.sum(sample_mean_adj[:,lag:] * sample_mean_adj[:,:-lag],axis=-1)
    sample_acf = self_cov / (sample_var * (min_pc_arr.shape[-1]) + 1e-8)
    return sample_acf

def batch_selected_acf(min_pc_arr, selection:list=[1,5,15,30,60]):
    """Caculate the ACF of a day on a batch of price changes series, given a set of lags.

    Input shape: (N, seq_len) where seq_len is the number of observations
    Output shape: list of len(selection) where each element is of shape (N,)
    """
    return [batch_acf(min_pc_arr, i) for i in selection]

def single_vol_clust(min_pc_arr, lag=5):
    """Caculate the degree of volatility clustering of a day on one price changes series, given one specific lag

    Input shape: (seq_len, ) where seq_len is the number of observations
    Output shape: (1,)
    """
    min_pc_arr_pow = min_pc_arr**2
    result = pearsonr(min_pc_arr_pow[:-lag], min_pc_arr_pow[lag:])
    return result.correlation

def batch_vol_clust(min_pc_arr, lag=1):
    """Caculate the degree of volatility clustering of a day on a batch of price changes series, given one specific lag

    Input shape: (N, seq_len) where seq_len is the number of observations
    Output shape: (N,)
    """
    pc_pow = min_pc_arr**2
    pc_pow_self = pc_pow[:,lag:]  # the sample part
    pc_pow_lag = pc_pow[:,:-lag]  # the lagged part
    std_pow_self = (pc_pow_self - pc_pow_self.mean(axis=-1, keepdims=True)) / (pc_pow_self.std(axis=-1, keepdims=True) + 1e-8)
    std_pow_lag = (pc_pow_lag - pc_pow_lag.mean(axis=-1, keepdims=True)) / (pc_pow_lag.std(axis=-1, keepdims=True) + 1e-8)
    clust = np.sum(std_pow_self * std_pow_lag, axis=-1) / pc_pow_self.shape[-1]
    return clust

def batch_selected_vol_clust(min_pc_arr, selection:list=[1,5,15,30,60]):
    """Caculate the degree of volatility clustering of a day on a batch of price changes series, given a set of lags

    Input shape: (N, seq_len) where seq_len is the number of observations
    Output shape: list of len(selection) where each element is of shape (N,)
    """
    return [batch_vol_clust(min_pc_arr, i) for i in selection]

def get_metrics_func(target):
    target_dict = {
        'return': batch_return,
        'amplitude': batch_amplitude,
        'volatility': batch_volatility,
        'skewness': batch_skew,
        'kurtosis': batch_kurtosis
    }
    return target_dict[target]
