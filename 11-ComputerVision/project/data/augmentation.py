import random
import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


def batch_transform(img_h, img_w):
    transform = A.Compose([
        A.Resize(height=img_h, width=img_w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return transform


def rand_bbox(size, lam):
    W = size[1]
    H = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(image1, image2, label1, label2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image1.shape, lam)
    
    image1[bbx1:bbx2, bby1:bby2, :] = image2[bbx1:bbx2, bby1:bby2, :]
    label = lam * label1 + (1 - lam) * label2
    
    return image1, label


def mixup(image1, image2, label1, label2, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    
    mixup_image = lam * image1 + (1 - lam) * image2
    mixup_image = mixup_image.astype(np.uint8)
    
    mixup_label = lam * label1 + (1 - lam) * label2
    
    return mixup_image, mixup_label


def cutout(image, mask_color=(0, 0, 0)):
    h, w, _ = image.shape
    mask_size = (h // random.randint(4, 6))

    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    cutout_image = image.copy()

    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)

    cutout_image[top:bottom, left:right] = mask_color

    return cutout_image


def quarter_divide(image):
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2

    top_left = image[0:center_y, 0:center_x]
    top_right = image[0:center_y, center_x:width]
    bottom_left = image[center_y:height, 0:center_x]
    bottom_right = image[center_y:height, center_x:width]

    results = [top_left, top_right, bottom_left, bottom_right]
    
    return results[random.randint(0, 3)]


def half_divide(image):
    height, width, _ = image.shape    
    center_y = height // 2

    top_half = image[0:center_y, :]
    bottom_half = image[center_y:height, :]

    results = [top_half, bottom_half]
    idx = random.randint(0, 1)
    
    return results[idx]


def divide_three_parts(image):
    height, width = image.shape[:2]

    part_height = height // 3
    top_part = image[:part_height, :]
    middle_part = image[part_height:2*part_height, :]
    bottom_part = image[2*part_height:, :]

    results = [top_part, middle_part, bottom_part]
    idx = random.randint(0, 2)

    return results[idx]


def divide_six_parts(image):
    height, width = image.shape[:2]

    block_height = height // 2
    block_width = width // 3

    block_1 = image[:block_height, :block_width]
    block_2 = image[:block_height, block_width:2*block_width]
    block_3 = image[:block_height, 2*block_width:]
    block_4 = image[block_height:, :block_width]
    block_5 = image[block_height:, block_width:2*block_width]
    block_6 = image[block_height:, 2*block_width:]

    results = [block_1, block_2, block_3, block_4, block_5, block_6]
    idx = random.randint(0, 5)

    return results[idx]


class QuarterDivide(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply=always_apply, p=p)
    
    def apply(self, img, **params):
        return quarter_divide(img)
    

class HalfDivide(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return half_divide(img)
    

class DivideThreeParts(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return divide_three_parts(img)
    

class DivideSixParts(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return divide_six_parts(img)