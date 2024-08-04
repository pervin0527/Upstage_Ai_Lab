import os
import timm
import torch
import argparse

from tqdm import tqdm
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score

from transformers import AutoImageProcessor, AutoModelForImageClassification

from loss import FocalLoss
from data.dataset import DocTypeDataset
from data.augmentation import batch_transform

from utils.data_util import train_valid_split
from utils.config_util import load_config, save_config
from utils.train_util import set_seed, make_save_dir, save_batch_images, CosineAnnealingWarmUpRestarts

def valid(model, dataloader, loss_func, device, writer, epoch, is_onehot):
    model.eval()
    valid_loss = 0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for _, images, labels in tqdm(dataloader, desc="Valid", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images).logits
            loss = loss_func(preds, labels)
            valid_loss += loss.item() * images.size(0)

            if not is_onehot:
                preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
                targets_list.extend(labels.detach().cpu().numpy())
            else:
                preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
                targets_list.extend(labels.argmax(dim=1).detach().cpu().numpy())

    valid_loss /= len(dataloader.dataset)
    valid_acc = accuracy_score(targets_list, preds_list)
    valid_f1 = f1_score(targets_list, preds_list, average='macro')

    writer.add_scalars('valid', {'Loss': valid_loss}, epoch)
    writer.add_scalars('valid', {'Accuracy': valid_acc}, epoch)
    writer.add_scalars('valid', {'F1_Score': valid_f1}, epoch)

    result = {
        "valid_loss": valid_loss,
        "valid_acc": valid_acc,
        "valid_f1": valid_f1,
    }

    return result


def train(model, dataloader, optimizer, loss_func, device, writer, epoch, is_onehot, gradient_accumulation_steps):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []
    optimizer.zero_grad()

    for step, (_, images, labels) in enumerate(tqdm(dataloader, desc="Train", leave=False)):
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images).logits
        loss = loss_func(preds, labels)
        loss = loss / gradient_accumulation_steps

        loss.backward()
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item() * images.size(0) * gradient_accumulation_steps

        if not is_onehot:
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(labels.detach().cpu().numpy())
        else:
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
            targets_list.extend(labels.argmax(dim=1).detach().cpu().numpy())

    train_loss /= len(dataloader.dataset)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    writer.add_scalars('train', {'Loss': train_loss}, epoch)
    writer.add_scalars('train', {'Accuracy': train_acc}, epoch)
    writer.add_scalars('train', {'F1_Score': train_f1}, epoch)

    result = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }

    return result


def main(cfg):
    save_dir = make_save_dir(cfg['save_path'])
    writer = SummaryWriter(log_dir=f"{save_dir}/logs")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = DocTypeDataset(cfg['train_img_path'], cfg['train_csv_path'], cfg['meta_path'], batch_transform(cfg['img_h'], cfg['img_w']), cfg['one_hot_encoding'])
    valid_dataset = DocTypeDataset(cfg['valid_img_path'], cfg['valid_csv_path'], cfg['meta_path'], batch_transform(cfg['img_h'], cfg['img_w']), cfg['one_hot_encoding'])
    print(f"Total train : {len(train_dataset)}, Total Valid : {len(valid_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'], shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    classes = train_dataset.classes

    if cfg['save_batch_imgs']:
        for batch_idx, data in enumerate(train_dataloader):
            _, images, labels = data
            print(images.shape, labels.shape)
            save_batch_images(data, output_dir=f"{save_dir}/batch_images")

            break

    # transformers 모델과 프로세서 로드
    processor = AutoImageProcessor.from_pretrained("amaye15/SwinV2-Base-Document-Classifier")
    model = AutoModelForImageClassification.from_pretrained("amaye15/SwinV2-Base-Document-Classifier", 
                                                            ignore_mismatched_sizes=True,
                                                            num_labels=len(classes)).to(device)

    if not cfg['focal_loss']:
        if not cfg['one_hot_encoding']:
            print("Loss function : CrossEntropy")
            loss_func = nn.CrossEntropyLoss()
        else:
            print("Loss function : BCEWithLogitsLoss")
            loss_func = nn.BCEWithLogitsLoss()
    else:
        print("Loss function : FocalLoss")
        loss_func = FocalLoss(cfg['one_hot_encoding'], alpha=cfg['focal_alpha'], gamma=cfg['focal_gamma'])

    optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=cfg['T_0'], T_mult=cfg['T_mult'], eta_max=cfg['max_lr'],  T_up=cfg['warmup_epochs'], gamma=cfg['T_gamma'])

    save_config(cfg, save_dir)
    best_f1_score = 0
    early_stopping_counter = 0
    early_stopping_patience = cfg['early_stop_patience']
    for epoch in range(1, cfg['epochs'] + 1):
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, epoch)
        print(f"Epoch [{epoch} | {cfg['epochs']}], LR : {current_lr}")
        
        train_result = train(model, train_dataloader, optimizer, loss_func, device, writer, epoch, cfg['one_hot_encoding'], cfg['gradient_accumulation_steps'])
        print(f"Train Loss : {train_result['train_loss']:.4f}, Train Acc : {train_result['train_acc']:.4f}, Train F1 : {train_result['train_f1']:.4f}")

        valid_result = valid(model, valid_dataloader, loss_func, device, writer, epoch, cfg['one_hot_encoding'])
        print(f"Valid Loss : {valid_result['valid_loss']:.4f}, Valid Acc : {valid_result['valid_acc']:.4f}, Valid F1 : {valid_result['valid_f1']:.4f}")

        scheduler.step()
        if valid_result['valid_f1'] > best_f1_score:
            print(f"Valid F1 Updated | prev : {best_f1_score:.4f} --> cur : {valid_result['valid_f1']:.4f}")
            best_f1_score = valid_result['valid_f1']
            torch.save(model.state_dict(), os.path.join(save_dir, 'weights', 'best.pth'))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Valid F1 Not Updated | early_stop_counter : {early_stopping_counter}")

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping")
            break

        print()

    torch.save(model.state_dict(), os.path.join(save_dir, 'weights', 'last.pth'))
    writer.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument('--config_path', type=str, default='./config.yaml', help='Path to the config file')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    cfg = load_config(config_path)

    set_seed(cfg['seed'])
    main(cfg)
