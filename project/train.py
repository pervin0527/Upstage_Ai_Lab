import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import json
import yaml
import torch
import wandb
import argparse
import pandas as pd
import pytorch_lightning as pl

from glob import glob
from tqdm import tqdm
from pprint import pprint
from rouge import Rouge

from transformers import EarlyStoppingCallback
from torch.utils.data import Dataset , DataLoader
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from utils.config_utils import load_config, save_config

def parse_args():
    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument('--config_path', type=str, default='./config.yaml', help='Path to the config file')
    args = parser.parse_args()

    return args

def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")
    
    data_path = cfg['general']['data_path']
    train_df = pd.read_csv(f"{data_path}/cleaned_train.csv")
    valid_df = pd.read_csv(f"{data_path}/cleaned_dev.csv")



if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    cfg = load_config(config_path)

    # pprint(cfg)
    main(cfg)