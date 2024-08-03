import os
import math
import torch
import random
import numpy as np
from datetime import datetime
from torch.backends import cudnn
from torchvision.utils import save_image
from torch.optim.lr_scheduler import _LRScheduler

def set_seed(seed_num):
    random.seed(seed_num)
    np.random.seed(seed_num)

    torch.manual_seed(seed_num) ## pytorch seed 설정(gpu 제외)
    
    torch.cuda.manual_seed(seed_num) ## pytorch cuda seed 설정
    torch.cuda.manual_seed_all(seed_num)

    cudnn.benchmark = False ## cudnn 커널 선정하는 과정을 허용하지 않는다.
    cudnn.deterministic = True ## 결정론적(동일한 입력에 대한 동일한 출력)으로 동작하도록 설정.


def make_save_dir(save_path):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(save_path, timestamp)
    os.makedirs(f"{save_dir}/weights", exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)

    return save_dir


def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # 반대로 정규화를 되돌림
    return tensor


def save_batch_images(data, output_dir="output_images"):
    os.makedirs(output_dir, exist_ok=True)

    ids, images, labels = data
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for idx, (id, image, label) in enumerate(zip(ids, images, labels)):
        img = denormalize(image, mean, std)        
        img = img.clamp(0, 1)

        image_filename = os.path.join(output_dir, id)
        save_image(img, image_filename)
        # print(f"Saved {image_filename}")


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr