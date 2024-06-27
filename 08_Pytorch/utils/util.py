import os
import yaml
from datetime import datetime


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return config


def save_config(cfg, save_dir):
    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)
    print(f"Configuration saved to {config_path}")


def mk_savedir(path):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(path, timestamp)
    log_dir = os.path.join(save_dir, 'logs')
    weight_dir = os.path.join(save_dir, 'weights')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_dir, exist_ok=True)
    
    return save_dir