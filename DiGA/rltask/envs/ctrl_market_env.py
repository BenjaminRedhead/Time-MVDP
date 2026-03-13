import numpy as np

import pandas as pd
from dateutil.parser import parse

from mlib.core.env import Env
from mlib.core.event import create_exchange_events
from mlib.core.exchange import Exchange

from agent.utils.chartist_state import ChartistState
from agent.rl_agent import RLAgent
from agent.utils.trade_info_state import TradeInfoState
from agent.utils.rl_state import RLState
from agent.meta_agent import get_oracle, parse_config, CtrlHeterogeneousAgent
from utils.pkl_utils import load_pkl
from rltask.envs.base_market_env import BaseMarketEnv
from mlib.core.exchange_config import create_exchange_config_without_call_auction # create_a_stock_exchange_config

base_ctrl_config = {
    "f_value": "g",
    "symbol": 'ma.sim',
    "init_price": 100000,
    "date": "20000101",
    "start_time": "09:30:00",
    "end_time": "15:00:00",
    "tick_size": 100,
    "num_agents": 400,
    "wakeup_prob": 0.01,
    "delta_time": 1,
    "freq": 1,
    "f": 10,
    "c": 1.5,
    "n": 1,
    "time_ref": 30,
    "init_cash": 25,
    "init_hold": 25,
    "a": 0.1,
    "risk_averse": 0.1,
    "min_num_returns": 20,
    "noise_var": 1e-4,
    "reward_mode": "step",
    "window": 30,
    "ctrl_path": "/data/container/rl_simulation/ctrl_env_train_dict.pkl",
    "shuffle": True
}

class CtrlMarketEnv(BaseMarketEnv):
    """Custom Environment that follows gym interface."""

    def __init__(self, config, mode='normal', show_progress=True, discrete_action=True):
        super().__init__(config=config, mode=mode, show_progress=show_progress, discrete_action=discrete_action)
        self.current_ctrl_idx = 0
        self.all_ctrl_pairs = None
        self.prepare_ctrl_data()

    def prepare_ctrl_data(self):
        assert 'ctrl_path' in self.config, "Control path not specified!"
        self.all_ctrls = load_pkl(self.config['ctrl_path'])
        self.all_ctrl_names = list(self.all_ctrls.keys())
        self.num_ctrls = len(self.all_ctrl_names)

    def get_next_ctrl_data(self):
        assert self.all_ctrls is not None, "Control data not prepared!"
        if self.config["shuffle"]:
            self.select_ctrl = np.random.choice(self.all_ctrl_names)
        else:
            self.select_ctrl = self.all_ctrl_names[self.current_ctrl_idx]
            self.current_ctrl_idx += 1
            if self.current_ctrl_idx >= self.num_ctrls:
                self.current_ctrl_idx = 0
        return self.all_ctrls[self.select_ctrl]

    def prepare_trading_env(self, config):
        """Run a rollout and get trade info."""
        next_ctrl = self.get_next_ctrl_data()
        date = pd.to_datetime(config["date"])  
        mkt_open: pd.Timestamp = date + pd.to_timedelta(parse(config["start_time"]).strftime("%H:%M:%S"))  # type: ignore
        mkt_close: pd.Timestamp = date + pd.to_timedelta(parse(config["end_time"]).strftime("%H:%M:%S"))
        self.close_time = mkt_close
        symbol = config['symbol']
        print(
            f"Reset exchange and environments...Symbol {symbol}, Data {date}"
        )
        ex_config = create_exchange_config_without_call_auction(mkt_open, mkt_close, [symbol])
        self.exchange = Exchange(ex_config)
        self.exchange.register_state(TradeInfoState())  
        self.rl_state = RLState(config['window'], config['init_price'], config['tick_size'])
        self.exchange.register_state(self.rl_state)
        self.chartist_state = ChartistState(config['delta_time'], config['min_num_returns'], config['init_price'])
        self.exchange.register_state(self.chartist_state)

        self.market_env = Env(self.exchange, "Market Env For RL", show_progress=self.show_progress)
        agent_config = parse_config(config, symbol, mkt_open, mkt_close)
        oracle, data_loader = get_oracle(agent_config=config, ctrls=next_ctrl, symbol=symbol, output_dir=None, mkt_open=mkt_open, mkt_close=mkt_close)
        background_agents = [
            CtrlHeterogeneousAgent(0, agent_config, chartist_state=self.chartist_state, oracle=oracle, symbol=config['symbol'])
        ]
        for bg_agent in background_agents:
            self.market_env.register_agent(bg_agent)
        self.rl_agent = RLAgent(start_time=mkt_open, end_time=mkt_close, obs_state=self.rl_state, symbol=config['symbol'])
        self.market_env.register_agent(self.rl_agent)
        self.market_env.push_events(create_exchange_events(ex_config))
        self.obs_generator = self.market_env.env()
        observation = self.market_loop()

        return observation
