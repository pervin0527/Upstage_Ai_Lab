import os
import torch
import random
import numpy as np
from datetime import datetime
from torch.backends import cudnn
from torchvision.utils import save_image

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