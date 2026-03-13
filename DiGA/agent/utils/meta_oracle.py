import os
import numpy as np
import pandas as pd

class CtrlOracle():
    def __init__(self, mkt_open, mkt_close, symbols, freq="S"):
        self.freq = freq
        self.mkt_close = mkt_close
        self.mkt_open = mkt_open
        self.symbols = symbols

        # The dictionary r holds the fundamenal value series for each symbol.
        self.r = {}
        # The dictionary n holds the number of orders for each symbol.
        self.n = {}
        for symbol in symbols:
            s = symbols[symbol]
            self.r[symbol] = self.generate_fundamental_value_series(symbol=symbol, **s)
            self.n[symbol] = s["dataLoader"].getNOrders().set_index("ts")['n_orders']
        self.f_log = {}
        for symbol in symbols:
            self.f_log[symbol] = [{"FundamentalTime": mkt_open, "FundamentalValue": self.r[symbol][0]}]


    def generate_fundamental_value_series(self, symbol, a, freq, dataLoader, save_pth):
        # Input: millisecond-wise mid-price
        # Output: second-wise mid-price sequence
        midPriceTb = dataLoader.getMidPrice()
        midPriceTb["ts"] = pd.to_datetime(midPriceTb["ts"], format=("%Y-%m-%d %H:%M:%S"))

        # Create the time series into which values will be projected and initialize the first value.
        date_range = pd.date_range(self.mkt_open, self.mkt_close, freq=self.freq)

        s = pd.Series(index=date_range)
        gt_df = pd.DataFrame(index=date_range)
        gt_df['price'] = midPriceTb.set_index('ts')['price']
        r_gt = gt_df['price'].fillna(method='ffill')

        duration = (self.mkt_close - self.mkt_open).total_seconds() / 60
        seg_num = int(duration / freq)
        r = seed_exp_func(r_gt, a, seg_num)
        # r = seed_linear(r)

        r_gt[:] = np.around(r_gt)
        r_gt = r_gt.astype(int)

        s[:] = np.round(r)
        s = s.astype(int)

        if save_pth:
            if not os.path.exists(save_pth):
                os.makedirs(save_pth)
            oracle_path = os.path.join(save_pth, "oracle_price.csv")
            s_cpy = {"ts": s.index, "price": r_gt}
            s_cpy = pd.DataFrame(s_cpy)
            s_cpy.to_csv(oracle_path, index=False)
            print("Mid-price ground truth has been saved!")

            data = pd.DataFrame({"FundamentalTime": r.index, "FundamentalValue": r})
            self.save_fundamental(data, os.path.join(save_pth, f"fundamental_{symbol}.bz2"))
        return s

    def save_fundamental(self, data: pd.DataFrame, path):
        data.rename(columns={"price": "FundamentalValue", "ts": "FundamentalTime"}, inplace=True)
        data.set_index("FundamentalTime", inplace=True)
        data = data.sample(int(len(data) * 0.99))
        data.to_pickle(path)

    def observePrice(self, symbol, currentTime, random_state=None):
        # If the request is made after market close, return the close price.
        # This function returns seed while saving ground truth for visualizatiion

        if currentTime >= self.mkt_close:
            r_t = self.r[symbol].loc[self.mkt_close - pd.Timedelta("1s")]
        else:
            if currentTime not in self.r[symbol]:
                delta = currentTime.microsecond / 1000000
                preTime = currentTime.floor('s') #  (currentTime - pd.Timedelta(seconds=delta))
                aftTime = preTime + pd.Timedelta("1s")
                if aftTime >= self.mkt_close:
                    aftTime = self.mkt_close - pd.Timedelta("1s")
                r_t = int((self.r[symbol].loc[aftTime] - self.r[symbol].loc[preTime]) * delta + self.r[symbol].loc[preTime])
            else:
                r_t = self.r[symbol].loc[currentTime]

        obs = seed_origin_value(r_t)

        # Reminder: all simulator prices are specified in integer cents.
        return obs

    def observeWakeupProb(self, symbol, currentTime, n_agents):

        if currentTime >= self.mkt_close:
            n_t = self.n[symbol].loc[self.mkt_close - pd.Timedelta("1s")]
        else:
            if currentTime not in self.n[symbol]:
                delta = currentTime.microsecond / 1000000
                preTime = currentTime - pd.Timedelta("{}s".format(delta))
                aftTime = preTime + pd.Timedelta("1s")
                if aftTime >= self.mkt_close:
                    aftTime = self.mkt_close - pd.Timedelta("1s")
                n_t = (self.n[symbol].loc[aftTime] - self.n[symbol].loc[preTime]) * delta + self.n[symbol].loc[preTime]
            else:
                n_t = self.n[symbol].loc[currentTime]

        prob = n_t / n_agents
        obs = seed_origin_value(prob)

        if self.f_log[symbol][-1]["FundamentalTime"] != currentTime:
            self.f_log[symbol].append({"FundamentalTime": currentTime, "WakeupProb": prob})

        # Reminder: all simulator prices are specified in integer cents.
        return obs

    def scheduleWakeUpTime(self, symbol):

        norders = self.n[symbol]
        this_time: pd.Timestamp = self.mkt_open
        wakeup_schedule = [this_time]
        while this_time < self.mkt_close:
            if ((this_time.hour==11 and this_time.minute>=30) or this_time.hour==12):
                this_time = pd.Timestamp(year=this_minute.year, month=this_minute.month, day=this_minute.day, hour=13, minute=0)
            this_minute = this_time.floor('min')

            this_lambda = norders.loc[this_minute]
            this_interval = np.random.exponential(1 / this_lambda)
            this_time += pd.Timedelta(minutes=this_interval)
            if this_time < self.mkt_close and not ((this_time.hour==11 and this_time.minute>=30) or this_time.hour==12):
                wakeup_schedule.append(this_time)

        return wakeup_schedule

    def generateSeed(self):
        seeds = {}
        for symbol in self.symbols:
            s = self.symbols[symbol]
            seeds[symbol] = s["dataLoader"].generateSeed()
        return seeds


def seed_origin_value(value):
    return value


def seed_exp_func(r, a, num_seg):
    res_price = []

    time = r.index
    price = list(r)

    price_len = len(price)
    seg_len = int(price_len / num_seg)
    sample_point = [int(seg_idx * seg_len) for seg_idx in range(num_seg)]
    sample_point.append(int(price_len - 1))

    for seg in range(num_seg):
        idx_start = sample_point[seg]
        idx_end = sample_point[seg + 1]
        idx_mid = int((idx_start + idx_end) / 2)

        price_start = price[idx_start]
        price_end = price[idx_end]
        price_mid = (price_start + price_end) / 2
        if price_end >= price_start:
            tmp_idx = np.arange(-1, 0, 1 / (idx_mid - idx_start))
            increase = -((-tmp_idx) ** a)
            increase = increase - increase.min()
            res_price += list((price_mid - price_start) * increase + price_start)

            tmp_idx = np.arange(0, 1, 1 / (idx_end - idx_mid))
            increase = tmp_idx**a
            res_price += list((price_end - price_mid) * increase + price_mid)
        else:
            tmp_idx = np.arange(-1, 0, 1 / (idx_mid - idx_start))
            increase = (-tmp_idx) ** a
            increase = increase - increase.max()
            res_price += list((price_start - price_mid) * increase + price_start)

            tmp_idx = np.arange(0, 1, 1 / (idx_end - idx_mid))
            increase = -(tmp_idx**a)
            res_price += list((price_mid - price_end) * increase + price_mid)
    res_price.append(price[-1])
    min_len = min(len(res_price), len(time))
    res = pd.Series(res_price[:min_len], index=time[:min_len])
    return res


def seed_linear(r):
    num_seg = 6
    res_price = []

    time = r.index
    price = list(r)

    price_len = len(price)
    seg_len = int(price_len / num_seg)
    sample_point = [int(seg_idx * seg_len) for seg_idx in range(num_seg)]
    sample_point.append(int(price_len - 1))

    for seg in range(num_seg):
        idx_start = sample_point[seg]
        idx_end = sample_point[seg + 1]

        price_start = price[idx_start]
        price_end = price[idx_end]

        if price_start <= price_end:
            increase = np.arange(0, 1, 1 / (idx_end - idx_start)) * (price_end - price_start) + price_start
        else:
            increase = -np.arange(0, 1, 1 / (idx_end - idx_start)) * (price_start - price_end) + price_start

        res_price += list(increase)

    res_price.append(price[-1])

    res = pd.Series(res_price, index=time)
    return res
