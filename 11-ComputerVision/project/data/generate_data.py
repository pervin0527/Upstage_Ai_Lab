import os
import cv2
import copy
import shutil
import random
import numpy as np
import pandas as pd
import albumentations as A

from PIL import Image
from tqdm import tqdm
from augraphy import *
from torchvision import transforms

from augmentation import QuarterDivide, HalfDivide, DivideThreeParts, DivideSixParts

def augraphy_transform():
    paper_phase = [
        OneOf([
            Moire(moire_density = (15,20),
                  moire_blend_method = "normal",
                  moire_blend_alpha = 0.1,
                  p=0.5),
            PatternGenerator(imgx=random.randint(256, 512),
                             imgy=random.randint(256, 512),
                             n_rotation_range=(10, 15),
                             color="random",
                             alpha_range=(0.25, 0.5),
                             p=0.5),
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
        A.OneOf([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.Transpose(p=0.3)
        ], p=0.45),

        A.OneOf([
            A.Compose([
                A.LongestMaxSize(max_size=max(img_h, img_w), p=1),
                A.PadIfNeeded(min_height=img_h, min_width=img_w, border_mode=0, value=(255, 255, 255), p=1),

                A.OneOf([
                    QuarterDivide(p=0.25),
                    HalfDivide(p=0.25),
                    DivideThreeParts(p=0.25),
                    DivideSixParts(p=0.25)
                ], p=0.5),

            ], p=0.25),

            A.ShiftScaleRotate(shift_limit_x=(-0.2, 0.2), 
                               shift_limit_y=(-0.2, 0.2), 
                               scale_limit=(-0.05, 0.05), 
                               rotate_limit=(-60, 60), 
                               interpolation=0, 
                               border_mode=0, 
                               value=(255, 255, 255),
                               rotate_method='largest_box',
                               p=0.25),

            A.Affine(keep_ratio=True, 
                     interpolation=0, 
                     rotate=(-45, 45), 
                     scale=(1.5, 2),
                     p=0.25),

            A.OpticalDistortion(distort_limit=(-0.3, 0.3), 
                                shift_limit=(-0.05, 0.05), 
                                interpolation=0, 
                                border_mode=0, 
                                value=(255, 255, 255), 
                                mask_value=None,
                                p=0.25)
        ], p=0.5),

        A.Resize(height=img_h, width=img_w, always_apply=True, p=1.0),
    ])

    return transform


def main():
    data_path = "../dataset"
    aug_iter = 50
    name = f"train_aug_{aug_iter}"
    save_path = f"{data_path}/{name}"
    img_h, img_w = 380, 380

    df = pd.read_csv(f"{data_path}/train.csv").sample(frac=1).reset_index(drop=True)
    meta_df = pd.read_csv(f"{data_path}/meta.csv")
    classes = list(meta_df['class_name'].unique())

    aup_transform = augraphy_transform()
    alb_transform = albumentation_transform(img_h, img_w)
    # torch_transform = transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)

    if os.path.exists("./augraphy_cache"):
        shutil.rmtree("./augraphy_cache")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    augmented_data = []
    os.makedirs(save_path, exist_ok=True)
    for idx1 in tqdm(range(0, len(df['ID']))):
        file_name = df.iloc[idx1, 0]
        target = df.iloc[idx1, 1]

        map_dict = {
            "45f0d2dfc7e47c03.jpg" : 7,
            "aec62dced7af97cd.jpg" : 14,
            "0583254a73b48ece.jpg" : 10,
            "1ec14a14bbe633db.jpg" : 7,
            "c5182ab809478f12.jpg" : 14,
            "8646f2c3280a4f49.jpg" : 3
        }
        if file_name in map_dict:
            target = map_dict[file_name]

        image_path = f"{data_path}/train/{file_name}"
        image = cv2.imread(image_path)
        cv2.imwrite(f"{save_path}/{file_name}", image)
        augmented_data.append({"ID": file_name, "target": target})

        for idx2 in tqdm(range(aug_iter), leave=False):
            ## augraphy + albumentations
            interm_img = aup_transform(copy.deepcopy(image))
            output1 = alb_transform(image=interm_img)['image']
            op1_name = f"aup_alb_{idx2:>03}_{file_name}"
            cv2.imwrite(f"{save_path}/{op1_name}", output1)
            augmented_data.append({"ID": op1_name, "target": target})

            ## only albumentations
            output2 = alb_transform(image=copy.deepcopy(image))['image']
            op2_name = f"alb_{idx2:>03}_{file_name}"
            cv2.imwrite(f"{save_path}/{op2_name}", output2)
            augmented_data.append({"ID": op2_name, "target": target})

            ## only augraphy
            output3 = aup_transform(copy.deepcopy(image))
            op3_name = f"aup_{idx2:>03}_{file_name}"
            cv2.imwrite(f"{save_path}/{op3_name}", output3)
            augmented_data.append({"ID": op3_name, "target": target})

    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(f"{data_path}/{name}.csv", index=False)

if __name__ == "__main__":
    main()
