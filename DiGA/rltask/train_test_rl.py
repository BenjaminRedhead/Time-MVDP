from datetime import datetime
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
from copy import copy
from pathlib import Path
from pytorch_lightning import seed_everything

from rltask.envs.ctrl_market_env import CtrlMarketEnv, base_ctrl_config
from rltask.envs.replay_market_env import ReplayMarketEnv, base_replay_config

def parse_args():
    parser = argparse.ArgumentParser(description='RL training')
    parser.add_argument('--market', type=str, default='DiGA', help='DiGA, Replay')
    parser.add_argument('--max_steps', type=int, default=288000, help='max_steps')
    parser.add_argument("--save_name", type=str, default="DiGA_env")
    parser.add_argument('--eval_eps', type=int, default=10, help='eval_eps')
    parser.add_argument('--data_path', type=str, default=None, help='path for diga/replay data')
    parser.add_argument('--test_replay_path', type=str, default=None, help='path for replay test data')
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument('--seed', type=int, default=0, help='seed')
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    seed_everything(args.seed)
    
    if args.market == 'DiGA':
        env_class = CtrlMarketEnv
        env_config = copy(base_ctrl_config)
        env_config['diga_path'] = args.data_path
    elif args.market == 'Replay':
        env_class = ReplayMarketEnv
        env_config = copy(base_replay_config)
        env_config['replay_path'] = args.data_path
    else:
        raise ValueError(f"Not included market type: {args.market}")
    
    model_class = A2C


    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{args.market}-disc-10s-w30-{args.max_steps}"
    if args.market == 'Ctrl':
        run_name = f"{args.market}-{args.ctrl_key}-disc-10s-w30-{args.max_steps}"
    run_name += f"-{args.seed}"
    save_path = Path(f"{args.output_path}/{args.save_name}/{run_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="Trade-Simulation-Market",
        config=args,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
        name=run_name
    )
    env_config['window'] = 30
    env = env_class(env_config, show_progress=False)
    env = Monitor(env)

    model = model_class("MlpPolicy", env, tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=args.max_steps, callback=WandbCallback(), progress_bar=True)
    model.save(Path(save_path) / "model") #
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=args.eval_eps)
    print(f"Train mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    env.close()


    env_config = copy(base_replay_config)
    env_config['train'] = False
    env_config['replay_path'] = args.test_replay_path
    env_config['window'] = 30
    env_config['test_pnl_path'] = save_path
    test_env = ReplayMarketEnv(env_config, show_progress=False)
    test_env = Monitor(test_env)

    model = model_class.load(Path(save_path)/run_name, env = test_env)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=args.eval_eps) 
    print(f"{run_name} Test mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    test_env.close()

# %%
