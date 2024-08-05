import os
import cv2
import copy
import shutil
import pandas as pd

from tqdm import tqdm
from augmentation import augraphy_transform, albumentation_transform

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
