from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG

from .types import Algorithm
from ..config import load_config
from ...types import Config


def parse_ppo_config(
    config: Config,
    default_config: Config = PPO_DEFAULT_CONFIG,
) -> Config:
    from core.utils.rl.skrl import populate_skrl_config

    final_config = default_config.copy()

    final_config.update(populate_skrl_config(config))

    return final_config


def load_and_parse_ppo_config(
    config_path: str,
    algo: Algorithm = Algorithm.PPO,
) -> Config:
    config = load_config(config_path)

    if algo == Algorithm.PPO:
        return parse_ppo_config(config)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
