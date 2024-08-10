import os
import cv2
import timm
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import DocTypeDataset
from utils.config_util import load_config
from utils.test_util import load_model, inference, save_predictions


def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = DocTypeDataset(cfg['test_img_path'], cfg['test_csv_path'], cfg['meta_path'], cfg['img_h'], cfg['img_w'], cfg["one_hot_encoding"], cfg['total_train'])
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    
    model = load_model(f"{cfg['saved_dir']}/weights/best.pth", cfg['model_name'], test_dataset.num_classes, device)
    ids, preds = inference(model, test_dataloader, device)

    save_predictions(ids, preds, test_dataset.classes, cfg)


def parse_args():
    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument('--saved_dir', type=str, default='./runs/best_9380', help='Path to Trained Dir')
    parser.add_argument('--img_h', type=int, default=0)
    parser.add_argument('--img_w', type=int, default=0)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    dir_path = args.saved_dir
    test_img_h = args.img_h
    test_img_w = args.img_w

    cfg = load_config(f"{dir_path}/config.yaml")
    cfg['saved_dir'] = dir_path

    if test_img_h > 0 and test_img_w > 0:
        cfg['img_h'] = test_img_h
        cfg['img_w'] = test_img_w

    main(cfg)