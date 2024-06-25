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
from models.autoencoder import AutoEncoder, Classifier
from utils.util import load_config, mk_savedir, save_config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train_autoencoder(autoencoder, optimizer, criterion, labeled_loader, unlabeled_loader, device, writer, epoch):
    autoencoder.train()
    total_labeled_loss = 0
    total_unlabeled_loss = 0

    # 라벨이 있는 데이터로 학습
    for data, _ in tqdm(labeled_loader, desc="Train AutoEncoder", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        recon, _ = autoencoder(data)
        loss = criterion(recon, data)
        loss.backward()
        optimizer.step()
        total_labeled_loss += loss.item()
    
    avg_labeled_loss = total_labeled_loss / len(labeled_loader)
    writer.add_scalar('Autoencoder/LabeledLoss', avg_labeled_loss, epoch)

    # 라벨이 없는 데이터로 학습
    for data, _ in tqdm(unlabeled_loader, desc="Train AutoEncoder", leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        recon, _ = autoencoder(data)
        loss = criterion(recon, data)
        loss.backward()
        optimizer.step()
        total_unlabeled_loss += loss.item()
    
    avg_unlabeled_loss = total_unlabeled_loss / len(unlabeled_loader)
    writer.add_scalar('Autoencoder/UnlabeledLoss', avg_unlabeled_loss, epoch)

    avg_loss = (total_labeled_loss + total_unlabeled_loss) / (len(labeled_loader) + len(unlabeled_loader))
    writer.add_scalar('Autoencoder/TotalLoss', avg_loss, epoch)

    return avg_labeled_loss, avg_unlabeled_loss, avg_loss


def test_autoencoder(autoencoder, test_loader, device, writer, epoch):
    autoencoder.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, _ = autoencoder(data)
            loss = criterion(recon, data)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    writer.add_scalar('Autoencoder/TestLoss', avg_loss, epoch)
    
    return avg_loss


def train_classifier(autoencoder, classifier, classifier_optim, criterion, labeled_loader, device, writer, epoch):
    classifier.train()
    total = 0
    correct = 0
    total_loss = 0
    for data, target in tqdm(labeled_loader, desc="Train Classifier", leave=False):
        data, target = data.to(device), target.to(device)
        classifier_optim.zero_grad()
        with torch.no_grad():
            _, latent = autoencoder(data)
        output = classifier(latent)
        loss = criterion(output, target)
        loss.backward()
        classifier_optim.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(labeled_loader)
    accuracy = 100. * correct / total
    writer.add_scalar('Classifier/Loss', avg_loss, epoch)
    writer.add_scalar('Classifier/Accuracy', accuracy, epoch)

    return avg_loss, accuracy


def test_classifier(autoencoder, classifier, test_loader, device, writer, epoch):
    classifier.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, latent = autoencoder(data)
            output = classifier(latent)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100. * correct / total
    writer.add_scalar('Classifier/TestAccuracy', accuracy, epoch)

    return accuracy


def main():
    cfg = load_config(config_path="./autoencoder_config.yaml")

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

    labeled_idx = np.random.choice(len(train_dataset), size=6000, replace=False)
    unlabeled_idx = list(set(range(len(train_dataset))) - set(labeled_idx))
    labeled_subset = torch.utils.data.Subset(train_dataset, labeled_idx)
    unlabeled_subset = torch.utils.data.Subset(train_dataset, unlabeled_idx)

    train_labeled_loader = DataLoader(labeled_subset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    train_unlabeled_loader = DataLoader(unlabeled_subset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])

    autoencoder = AutoEncoder(latent_dim=cfg['latent_dim']).to(device)
    autoencoder_optim = torch.optim.Adam(autoencoder.parameters(), lr=cfg['autoencoder_lr'], weight_decay=cfg['autoencoder_weight_decay'])
    autoencoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(autoencoder_optim, mode='min', patience=cfg['lr_patience'], factor=cfg['lr_factor'], verbose=True)
    autoencoder_cost = torch.nn.MSELoss()

    classifier = Classifier(input_dim=cfg['latent_dim'], num_classes=10).to(device)
    classifier_optim = torch.optim.Adam(classifier.parameters(), lr=cfg['classifier_lr'], weight_decay=cfg['classifier_weight_decay'])
    classifier_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifier_optim, mode='max', patience=cfg['lr_patience'], factor=cfg['lr_factor'], verbose=True)
    classifier_cost = torch.nn.CrossEntropyLoss()

    best_autoencoder_loss = float('inf')
    autoencoder_patience = cfg['earlystop_patience']
    autoencoder_counter = 0
    for epoch in range(1, cfg['autoencoder_epochs'] + 1):
        print(f"Epoch : {epoch}")
        labeled_loss, unlabeled_loss, train_loss = train_autoencoder(autoencoder, autoencoder_optim, autoencoder_cost, train_labeled_loader, train_unlabeled_loader, device, writer, epoch)
        test_loss = test_autoencoder(autoencoder, test_loader, device, writer, epoch)

        print(f'Train Loss: {train_loss:.4f}, Labeled Loss: {labeled_loss:.4f}, Unlabeled Loss: {unlabeled_loss:.4f}')
        print(f'Test Loss: {test_loss:.4f}')
        
        autoencoder_scheduler.step(test_loss)
        
        if test_loss < best_autoencoder_loss:
            best_autoencoder_loss = test_loss
            torch.save(autoencoder.state_dict(), os.path.join(save_dir, 'autoencoder_best.pth'))
            autoencoder_counter = 0
        else:
            autoencoder_counter += 1
            if autoencoder_counter >= autoencoder_patience:
                print("Early stopping for Autoencoder!")
                break
        
        print(f"Early Stop Counter : {autoencoder_counter} \n")
    
    torch.save(autoencoder.state_dict(), os.path.join(save_dir, 'autoencoder_last.pth'))

    best_classifier_accuracy = 0
    classifier_counter = 0
    classifier_patience = cfg['earlystop_patience']
    for epoch in range(1, cfg['classifier_epochs'] + 1):
        print(f"Epoch : {epoch}")
        classifier_train_loss, classifier_train_accuracy = train_classifier(autoencoder, classifier, classifier_optim, classifier_cost, train_labeled_loader, device, writer, epoch)
        classifier_test_accuracy = test_classifier(autoencoder, classifier, test_loader, device, writer, epoch)

        print(f'Train Loss: {classifier_train_loss:.4f}, Train Accuracy: {classifier_train_accuracy:.4f}')
        print(f'Test Accuracy: {classifier_test_accuracy:.4f} \n')
        
        classifier_scheduler.step(classifier_test_accuracy)
        
        if classifier_test_accuracy > best_classifier_accuracy:
            best_classifier_accuracy = classifier_test_accuracy
            torch.save(classifier.state_dict(), os.path.join(save_dir, 'classifier_best.pth'))
            classifier_counter = 0
        else:
            classifier_counter += 1
            if classifier_counter >= classifier_patience:
                print("Early stopping for Classifier!")
                break
        print(f"Early Stop Counter : {classifier_counter} \n")

    torch.save(classifier.state_dict(), os.path.join(save_dir, 'classifier_last.pth'))
    writer.close()

if __name__ == "__main__":
    main()
