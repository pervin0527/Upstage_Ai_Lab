import os
import torch
import random
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import datasets
from torchvision.utils import make_grid
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

from data.dataset import ssl_preprocessing
from data.dataset import AlbumentationsDataset
from models.autoencoder import AutoEncoder, EncoderClassifier
from utils.util import load_config, mk_savedir, save_config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(model, dataloader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader, desc="Train AutoEncoder", leave=False):
        x = x.to(device)

        optimizer.zero_grad()
        y_pred = model(x)

        loss = criterion(y_pred, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    total_loss /= len(dataloader.dataset)
    writer.add_scalar('Loss/Train AutoEncoder', total_loss, epoch)

    return total_loss

def eval(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Eval AutoEncoder", leave=False):
            x = x.to(device)
            y_pred = model(x)

            loss = criterion(y_pred, x)
            total_loss += loss.item() * x.size(0)

            if x.shape[0] == dataloader.batch_size:
                orig_images = x.cpu()
                recon_images = y_pred.cpu()

                orig_grid = make_grid(orig_images, nrow=8, normalize=True)
                recon_grid = make_grid(recon_images, nrow=8, normalize=True)

                writer.add_image('Original Images', orig_grid, epoch)
                writer.add_image('Reconstructed Images', recon_grid, epoch)

                fig, axes = plt.subplots(1, 2, figsize=(12, 12))
                axes[0].imshow(orig_grid.permute(1, 2, 0).numpy())
                axes[0].set_title('Original Images')
                axes[0].axis('off')

                axes[1].imshow(recon_grid.permute(1, 2, 0).numpy())
                axes[1].set_title('Reconstructed Images')
                axes[1].axis('off')

                plt.show()
                break

    total_loss /= len(dataloader.dataset)
    writer.add_scalar('Loss/Eval AutoEncoder', total_loss, epoch)

    return total_loss

def train_classifier(model, dataloader, optimizer, criterion, device, writer, epoch):
    model.train()
    total_loss = 0
    total_correct = 0
    for x, y in tqdm(dataloader, desc='Train Classifier', leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == y).sum().item()

    total_loss /= len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)
    writer.add_scalar('Loss/Train Classifier', total_loss, epoch)
    writer.add_scalar('Accuracy/Train Classifier', accuracy, epoch)

    return total_loss, accuracy

def eval_classifier(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc='Eval Classifier', leave=False):
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            total_loss += loss.item() * x.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y).sum().item()

    total_loss /= len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)
    writer.add_scalar('Loss/Eval Classifier', total_loss, epoch)
    writer.add_scalar('Accuracy/Eval Classifier', accuracy, epoch)

    return total_loss, accuracy

def main():
    cfg = load_config(config_path="./autoencoder_config.yaml")

    save_dir = mk_savedir(cfg['save_path'])
    save_config(cfg, save_dir)

    set_seed(cfg['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=f"{save_dir}/logs")

    train_dataset = datasets.MNIST(cfg['dataset_path'], train=True, download=True, transform=None)
    test_dataset = datasets.MNIST(cfg['dataset_path'], train=False, download=True, transform=None)

    data_dict = ssl_preprocessing(train_dataset, test_dataset, num_labeled=cfg['num_labeled'], num_unlabeled=cfg['num_unlabeled'])
    labeled_train_dataset = TensorDataset(data_dict["x_train_labeled"], data_dict["y_train_labeled"])
    unlabeled_train_dataset = TensorDataset(data_dict["x_train_unlabeled"], torch.zeros(data_dict["x_train_unlabeled"].shape[0]))
    test_dataset = TensorDataset(data_dict["x_test"], data_dict["y_test"])

    labeled_train_dataloader = DataLoader(labeled_train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

    autoencoder = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=cfg['learning_rate'])
    criterion = torch.nn.MSELoss()

    for epoch in range(1, cfg['epochs']+1):
        print(f"AutoEncoder Epoch : {epoch}/{cfg['epochs']}")

        train_loss = train(autoencoder, unlabeled_train_dataloader, optimizer, criterion, device, writer, epoch)
        print(f"Train AutoEncoder Loss : {train_loss:.4f}")

        eval_loss = eval(autoencoder, test_dataloader, criterion, device, writer, epoch)
        print(f"Eval AutoEncoder Loss : {eval_loss:.4f}\n")


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

    classifier = EncoderClassifier(autoencoder, num_classes=10).to(device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg['learning_rate'])
    classifier_criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, cfg['epochs']+1):
        print(f"Classifier Epoch : {epoch}/{cfg['epochs']}")

        train_loss, train_accuracy = train_classifier(classifier, train_dataloader, classifier_optimizer, classifier_criterion, device, writer, epoch)
        print(f"Train Classifier Loss : {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

        eval_loss, eval_accuracy = eval_classifier(classifier, test_dataloader, classifier_criterion, device, writer, epoch)
        print(f"Eval Classifier Loss : {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}\n")

if __name__ == "__main__":
    main()
