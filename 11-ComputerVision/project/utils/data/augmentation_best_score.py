import random
import numpy as np
import albumentations as A

from augraphy import *
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform


def batch_transform(img_h, img_w):
    transform = A.Compose([
        A.Resize(height=img_h, width=img_w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return transform


def augraphy_transform():
    ink_phase = [
        BleedThrough(intensity_range=(0.1, 0.3),
                     color_range=(32, 224),
                     ksize=(17, 17),
                     sigmaX=1,
                     alpha=random.uniform(0.1, 0.2),
                     offsets=(10, 20),
                     p=0.4),
    ]

    paper_phase = [
        OneOf([
            Moire(moire_density = (15,20),
                  moire_blend_method = "normal",
                  moire_blend_alpha = 0.1,
                  p=0.35),

            PatternGenerator(imgx=random.randint(256, 512),
                             imgy=random.randint(256, 512),
                             n_rotation_range=(10, 15),
                             color="random",
                             alpha_range=(0.25, 0.5),
                             p=0.35),

            VoronoiTessellation(mult_range=(50,80),
                                seed=19829813472,
                                num_cells_range=(500,800),
                                noise_type="random",
                                background_value=(200, 256),
                                p=0.3),
        ], p=0.25),
        
        OneOf([
            ## 테스트 데이터 노이즈와 유사한듯??
            NoiseTexturize(sigma_range=(5, 15),
                           turbulence_range=(3, 9),
                           texture_width_range=(50, 500),
                           texture_height_range=(50, 500),
                           p=0.5),
            BrightnessTexturize(texturize_range=(0.8, 0.99), deviation=0.02, p=0.5),
        ], p=0.6),
    ]

    post_phase = [
        ## 부분 밝기, 그림자
        OneOf([
            LightingGradient(light_position=None,
                             direction=90,
                             max_brightness=255,
                             min_brightness=0,
                             mode="gaussian",
                             transparency=0.5,
                             p=0.25),

            LowLightNoise(num_photons_range=(50, 100), 
                          alpha_range=(0.7, 0.9), 
                          beta_range=(10, 30), 
                          gamma_range=(1.0 , 1.8),
                          p=0.25),

            ReflectedLight(reflected_light_smoothness = 0.8,
                           reflected_light_internal_radius_range=(0.0, 0.2),
                           reflected_light_external_radius_range=(0.1, 0.8),
                           reflected_light_minor_major_ratio_range = (0.9, 1.0),
                           reflected_light_color = (255,255,255),
                           reflected_light_internal_max_brightness_range=(0.9,1.0),
                           reflected_light_external_max_brightness_range=(0.9,0.9),
                           reflected_light_location = "random",
                           reflected_light_ellipse_angle_range = (0, 360),
                           reflected_light_gaussian_kernel_size_range = (5,310),
                           p=0.25),

            ShadowCast(shadow_side = "bottom",
                    shadow_vertices_range = (2, 3),
                    shadow_width_range=(0.5, 0.8),
                    shadow_height_range=(0.5, 0.8),
                    shadow_color = (0, 0, 0),
                    shadow_opacity_range=(0.5,0.6),
                    shadow_iterations_range = (1,2),
                    shadow_blur_kernel_range = (101, 301),
                    p=0.25)

        ], p=0.25),      
    ]
    
    return AugraphyPipeline(
        # pre_phase=[rescale],
        # ink_phase=ink_phase, 
        paper_phase=paper_phase, 
        post_phase=post_phase
    )


def albumentation_transform(img_h, img_w):
    transform = A.Compose([
        A.Compose([
            A.LongestMaxSize(max_size=max(img_h, img_w), p=1),
            A.PadIfNeeded(min_height=img_h, min_width=img_w, border_mode=0, value=(255, 255, 255), p=1),

            A.OneOf([
                QuarterDivide(p=0.2),
                HalfDivide(p=0.2),
                DivideThreeParts(p=0.2),
                DivideSixParts(p=0.2),
                A.RandomCrop(height=img_h//2, width=img_w//2, p=0.2)
            ], p=0.55),
        ], p=0.6),

        A.OneOf([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Transpose(p=0.4)
        ], p=0.6),

        A.OneOf([
            A.RandomRotate90(p=0.2),

            A.ShiftScaleRotate(shift_limit_x=(-0.2, 0.2), 
                               shift_limit_y=(-0.2, 0.2), 
                               scale_limit=(-0.05, 0.05), 
                               rotate_limit=(-60, 60), 
                               interpolation=0, 
                               border_mode=0, 
                               value=(255, 255, 255),
                               rotate_method='largest_box',
                               p=0.3),

            A.Affine(keep_ratio=True, 
                     interpolation=0, 
                     rotate=(-45, 45), 
                     scale=(1.5, 2),
                     p=0.3),

            A.OpticalDistortion(distort_limit=(-0.3, 0.3), 
                                shift_limit=(-0.05, 0.05), 
                                interpolation=0, 
                                border_mode=0, 
                                value=(255, 255, 255), 
                                mask_value=None,
                                p=0.2),
        ], p=0.6),

        A.Resize(height=img_h, width=img_w, always_apply=True, p=1.0),
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