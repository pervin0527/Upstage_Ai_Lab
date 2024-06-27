import os
import torch
import random
import numpy as np
import albumentations as A

from tqdm import tqdm
from torch.backends import cudnn
from torchvision import datasets
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

from data.dataset import AlbumentationsDataset
from models.cnn import CNN_V1, CNN_V2, CNN_V3, CNN_V4, CNN_V5
from models.rnn import RNN_V1, RNN_V2
from utils.util import load_config, mk_savedir, save_config

def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)
    
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    cudnn.benchmark = False
    cudnn.deterministic = True


def train(model, dataloader, criterion, optimizer, device, writer, epoch):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for x, y in tqdm(dataloader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        if isinstance(model, RNN_V1) or isinstance(model, RNN_V2):
            x = x.squeeze(1).reshape(-1, 28, 28)  ## RNN 모델일 경우 [batch_size, 28, 28]로 바꿔주고 행단위로 데이터를 입력.

        optimizer.zero_grad()
        y_pred = model(x)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)

        _, pred_labels = torch.max(y_pred, 1)
        train_acc += (pred_labels == y).sum().item()

    train_loss /= len(dataloader.dataset)
    train_acc /= len(dataloader.dataset)
    
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)

    return train_loss, train_acc


def eval(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    valid_loss, valid_acc = 0.0, 0.0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Valid", leave=False):
            x, y = x.to(device), y.to(device)
            if isinstance(model, RNN_V1) or isinstance(model, RNN_V2):
                x = x.squeeze(1).reshape(-1, 28, 28)
            y_pred = model(x)

            loss = criterion(y_pred, y)

            valid_loss += loss.item() * x.size(0)

            _, pred_labels = torch.max(y_pred, 1)
            valid_acc += (pred_labels == y).sum().item()

        valid_loss /= len(dataloader.dataset)
        valid_acc /= len(dataloader.dataset)
        
        writer.add_scalar('Loss/Test', valid_loss, epoch)
        writer.add_scalar('Accuracy/Test', valid_acc, epoch)

    return valid_loss, valid_acc


def final_test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Final Test", leave=False):
            x, y = x.to(device), y.to(device)
            if isinstance(model, RNN_V1) or isinstance(model, RNN_V2):
                x = x.squeeze(1).reshape(-1, 28, 28)
            y_pred = model(x)

            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    cfg = load_config(config_path="./classifier_config.yaml")

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

    if cfg['model'] == "CNN_V1":
        model = CNN_V1().to(device)
    elif cfg['model'] == "CNN_V2":
        model = CNN_V2().to(device)
    elif cfg['model'] == "CNN_V3":
        model = CNN_V3().to(device)
    elif cfg['model'] == 'CNN_V4':
        model = CNN_V4(drop_prob=cfg['drop_prob'], weight_init=cfg['weight_init']).to(device)
    elif cfg['model'] == 'CNN_V5':
        model = CNN_V5(drop_prob=cfg['drop_prob'], weight_init=cfg['weight_init']).to(device)
    elif cfg['model'] == 'RNN_V1':
        model = RNN_V1(input_size=28, hidden_size=cfg['hidden_dim'], output_size=10, num_layers=cfg['num_layers']).to(device)
    elif cfg['model'] == 'RNN_V2':
        model = RNN_V2(input_size=28, hidden_size=cfg['hidden_dim'], output_size=10, num_layers=cfg['num_layers']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=cfg['lr_patience'], factor=cfg['lr_factor'], verbose=True)
    criterion = torch.nn.CrossEntropyLoss()

    best_test_acc = 0
    early_stop_counter = 0
    for epoch in range(1, cfg['epochs']+1):
        print(f"Epoch : {epoch}")
        
        train_loss, train_acc = train(model, train_dataloader, criterion, optimizer, device, writer, epoch)
        print(f"Train Loss : {train_loss:.4f}, Train Acc : {train_acc:.4f}")

        test_loss, test_acc = eval(model, test_dataloader, criterion, device, writer, epoch)
        print(f"Test Loss : {test_loss:.4f}, Test Acc : {test_acc:.4f}")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(f"{save_dir}/weights", 'best.pth'))
        else:
            early_stop_counter += 1

        print(f"Early Stop Counter : {early_stop_counter}")

        scheduler.step(test_acc)
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
