import os
import torch
import random
import numpy as np
import albumentations as A

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

from data.dataset import AlbumentationsDataset
from models.autoencoder import AutoEncoder
from utils.util import load_config, mk_savedir, save_config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(model, dataloader, criterion_reconstruction, criterion_classification, optimizer, device, writer, epoch):
    model.train()
    train_loss_reconstruction, train_loss_classification, train_acc = 0.0, 0.0, 0.0
    for x, y in tqdm(dataloader, desc="Train", leave=False):
        x, y = x.view(x.size(0), -1).to(device), y.to(device)  # [batch_size, 784]로 변경
        optimizer.zero_grad()
        x_reconstructed, class_logits = model(x)

        loss_reconstruction = criterion_reconstruction(x_reconstructed, x)
        loss_classification = criterion_classification(class_logits, y)
        loss = loss_reconstruction + loss_classification
        loss.backward()
        optimizer.step()

        train_loss_reconstruction += loss_reconstruction.item() * x.size(0)
        train_loss_classification += loss_classification.item() * x.size(0)

        _, pred_labels = torch.max(class_logits, 1)
        train_acc += (pred_labels == y).sum().item()

    train_loss_reconstruction /= len(dataloader.dataset)
    train_loss_classification /= len(dataloader.dataset)
    train_acc /= len(dataloader.dataset)
    
    writer.add_scalar('Loss/Train/Reconstruction', train_loss_reconstruction, epoch)
    writer.add_scalar('Loss/Train/Classification', train_loss_classification, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)

    return train_loss_reconstruction, train_loss_classification, train_acc


def eval(model, dataloader, criterion_reconstruction, criterion_classification, device, writer, epoch):
    model.eval()
    valid_loss_reconstruction, valid_loss_classification, valid_acc = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Valid", leave=False):
            x, y = x.view(x.size(0), -1).to(device), y.to(device)
            x_reconstructed, class_logits = model(x)

            loss_reconstruction = criterion_reconstruction(x_reconstructed, x)
            loss_classification = criterion_classification(class_logits, y)

            valid_loss_reconstruction += loss_reconstruction.item() * x.size(0)
            valid_loss_classification += loss_classification.item() * x.size(0)

            _, pred_labels = torch.max(class_logits, 1)
            valid_acc += (pred_labels == y).sum().item()

        valid_loss_reconstruction /= len(dataloader.dataset)
        valid_loss_classification /= len(dataloader.dataset)
        valid_acc /= len(dataloader.dataset)
        
        writer.add_scalar('Loss/Test/Reconstruction', valid_loss_reconstruction, epoch)
        writer.add_scalar('Loss/Test/Classification', valid_loss_classification, epoch)
        writer.add_scalar('Accuracy/Test', valid_acc, epoch)

    return valid_loss_reconstruction, valid_loss_classification, valid_acc


def final_test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Final Test", leave=False):
            x, y = x.view(x.size(0), -1).to(device), y.to(device)
            _, class_logits = model(x)
            
            _, predicted = torch.max(class_logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    print(f'Final Test Accuracy: {accuracy:.2f}%')
    return accuracy


def main():
    cfg = load_config(config_path="./config.yaml")

    save_dir = mk_savedir(cfg['save_path'])
    save_config(cfg, save_dir)

    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=f"{save_dir}/logs")

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(p=0.4),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2()
    ])

    train_dataset = datasets.MNIST(cfg['dataset_path'], train=True, download=True, transform=None)
    train_dataset = AlbumentationsDataset(train_dataset, train_transform)

    test_dataset = datasets.MNIST(cfg['dataset_path'], train=False, download=True, transform=None)
    test_dataset = AlbumentationsDataset(test_dataset, test_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    model = AutoEncoder().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=cfg['lr_patience'], factor=cfg['lr_factor'], verbose=True)
    criterion_reconstruction = torch.nn.MSELoss()
    criterion_classification = torch.nn.CrossEntropyLoss()

    best_valid_acc = 0
    early_stop_counter = 0
    for epoch in range(1, cfg['epochs']+1):
        print(f"Epoch : {epoch}")
        
        train_loss_reconstruction, train_loss_classification, train_acc = train(model, train_dataloader, criterion_reconstruction, criterion_classification, optimizer, device, writer, epoch)
        print(f"Train Loss Reconstruction: {train_loss_reconstruction:.4f}, Train Loss Classification: {train_loss_classification:.4f}, Train Acc: {train_acc:.4f}")

        valid_loss_reconstruction, valid_loss_classification, valid_acc = eval(model, test_dataloader, criterion_reconstruction, criterion_classification, device, writer, epoch)
        print(f"Valid Loss Reconstruction: {valid_loss_reconstruction:.4f}, Valid Loss Classification: {valid_loss_classification:.4f}, Valid Acc: {valid_acc:.4f}")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(f"{save_dir}/weights", 'best.pth'))
        else:
            early_stop_counter += 1

        print(f"Early Stop Counter : {early_stop_counter}")

        scheduler.step(valid_acc)
        if early_stop_counter >= cfg['early_stop_patience']:
            print("Early stopping triggered")
            break
        print()

    torch.save(model.state_dict(), os.path.join(f"{save_dir}/weights", 'last.pth'))
    writer.close()

    final_accuracy = final_test(model, test_dataloader, device)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()
