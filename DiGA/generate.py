import argparse, os
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from dateutil.parser import parse

from mlib.core.engine import Engine
from mlib.core.event import create_exchange_events
from mlib.core.exchange import Exchange
from mlib.core.exchange_config import create_exchange_config_without_call_auction

from diffusion.cont_ctrl_net import UnetCC, DataModuleCC
from diffusion.disc_ctrl_net import UnetDC, DataModuleDC
from diffusion.ddpm import GaussianDiffusion, PLModel
from utils.pkl_utils import load_pkl, save_pkl
from agent.utils.chartist_state import ChartistState
from agent.utils.trade_info_state import TradeInfoState
from agent.meta_agent import CtrlHeterogeneousAgent, parse_config, base_config, get_oracle

def get_pseudo_price(args):
    if args.random_price:
        pseudo_init_price = (5 + np.random.exponential(scale=10)) * 10000 // 50 * 50
    else:
        pseudo_init_price = args.pseudo_price * 10000
    return pseudo_init_price


def _get_trade_infos(exchange: Exchange, symbol: str, start_time: pd.Timestamp, end_time: pd.Timestamp):
    state = exchange.states()[symbol][TradeInfoState.__name__]
    assert isinstance(state, TradeInfoState)
    trade_infos = state.trade_infos
    trade_infos = [x for x in trade_infos if start_time <= x.order.time <= end_time]
    return trade_infos

def _simulate(oracle, config):
    chartist_state = ChartistState(config.delta_time, config.min_num_returns, config.init_price)
    meta_agent = CtrlHeterogeneousAgent(0, config, chartist_state=chartist_state, oracle=oracle, symbol=config.symbol)

    exchange_config = create_exchange_config_without_call_auction(config.start_time, config.end_time, symbols=[config.symbol])

    exchange = Exchange(exchange_config)
    exchange.register_state(chartist_state)
    exchange.register_state(TradeInfoState())

    engine = Engine(exchange, "Meta-agent working")
    engine.register_agent(meta_agent)
    engine.push_events(create_exchange_events(exchange_config))
    engine.run()
    sim_trade_infos = _get_trade_infos(exchange, config.symbol, config.start_time, config.end_time)
    return sim_trade_infos

def run_ctrl_simulation(ctrls: Dict, configs, result_path_name='DiGA_generated_market', result_name='generation_0', agent_seed=None, symbol='sim_stock'):
    output_dir = Path(result_path_name) / result_name
    output_dir.mkdir(parents=True, exist_ok=True)  # copy json config to output dir
    historical_date: pd.Timestamp = pd.to_datetime(configs["date"])  # type: ignore
    mkt_open: pd.Timestamp = historical_date + pd.to_timedelta(parse(configs["start_time"]).strftime("%H:%M:%S"))  # type: ignore
    mkt_close: pd.Timestamp = historical_date + pd.to_timedelta(parse(configs["end_time"]).strftime("%H:%M:%S"))  # type: ignore

    oracle, data_loader = get_oracle(agent_config=configs, ctrls=ctrls, symbol=symbol, output_dir=output_dir, mkt_open=mkt_open, mkt_close=mkt_close)

    agent_config = parse_config(configs, symbol, mkt_open, mkt_close)

    if not isinstance(agent_seed, int):
        agent_seed = data_loader.generateSeed()
    np.random.seed(agent_seed)
    trade_info_path = output_dir / f"trade_infos_{result_name}-seed_{agent_seed}.pkl"
    if os.path.isfile(trade_info_path):
        print(f"Result exists! Skip {trade_info_path}")
        return load_pkl(trade_info_path)
    sim_trade_infos = _simulate(oracle=oracle, config=agent_config)
    save_pkl(sim_trade_infos, path=trade_info_path)
    print(f"Save {len(sim_trade_infos)} trade infos to {trade_info_path}")
    return sim_trade_infos

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="SZAMain")
    parser.add_argument("--ctrl_type", type=str, default="continuous")
    parser.add_argument("--ctrl_target", type=str, default="return")
    parser.add_argument("--ctrl_class", type=float, default=0)
    parser.add_argument("--cond_scale", type=float, default=1)
    parser.add_argument("--samsteps", type=int, default=20)
    parser.add_argument("--diffsteps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--save_name", type=str, default="DiGA_generation")
    parser.add_argument("--random_price", action="store_true")
    parser.add_argument("--pseudo_price", type=float, default=10)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()    
    
    seed_everything(args.seed)
    print(f"Set seed: {args.seed}")
    train_path = Path(args.data_path) / f'{args.data_name}_train.npy'
    vali_path = Path(args.data_path) / f'{args.data_name}_vali.npy'

    results_folder = Path(args.output_path) / args.exp_name
    print(f"Set output path: {results_folder}")
    
    if args.ctrl_type == 'continuous':
        pl_data = DataModuleCC(train_path, vali_path, selected_metrics=args.ctrl_target, n_bins=args.n_bins)
        model = UnetCC(dim = 64, n_ctrls = 1, dim_mults = (1, 4, 16), channels = 2)
        cond_class = float(args.ctrl_class)
    elif args.ctrl_type == 'discrete':
        pl_data = DataModuleDC(train_path, vali_path, selected_metrics=args.ctrl_target, n_bins=args.n_bins)
        model = UnetDC(dim = 64, num_classes=args.n_bins, dim_mults = (1, 4, 16), channels = 2)
        cond_class = int(args.ctrl_class)
        
    diffusion = GaussianDiffusion(model, seq_length = 236, timesteps = args.diffsteps, sampling_timesteps = args.samsteps, objective = 'pred_noise', auto_normalize = False)
    
    cp_path = results_folder / args.checkpoint_path
    LitModel = PLModel.load_from_checkpoint(cp_path, model=diffusion).cuda()
    print(f"Loaded from checkpoint: {cp_path}")
    LitModel.model.eval()

    samples = LitModel.model.sample(torch.tensor([[cond_class]]).to(LitModel.device), cond_scale=args.cond_scale).cpu().numpy()
    sample = pl_data.inverse_transform(samples)[0]
    sampled_price = np.exp((sample[0]).squeeze().cumsum(axis=-1)) * get_pseudo_price(args)
    sampled_no = sample[1]
    ctrl_dict = {
        'fundamental': sampled_price,
        'n_orders': sampled_no
    }
    
    trade_info = run_ctrl_simulation(ctrls=ctrl_dict, configs=base_config, result_path_name=results_folder, result_name=args.save_name, agent_seed=args.seed)
