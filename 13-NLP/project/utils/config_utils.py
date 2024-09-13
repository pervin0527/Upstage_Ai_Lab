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

def make_save_dir(save_path):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(save_path, timestamp)
    # os.makedirs(f"{save_dir}/weights", exist_ok=True)
    # os.makedirs(f"{save_dir}/logs", exist_ok=True)

    return save_dir