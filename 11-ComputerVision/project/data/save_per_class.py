import os
import cv2
import pandas as pd

def main():
    data_path = "../dataset"
    meta_df = pd.read_csv(f"{data_path}/meta.csv")
    classes = list(meta_df['class_name'].unique())

    file_df = pd.read_csv(f"{data_path}/train.csv")
    os.makedirs(f"{data_path}/images", exist_ok=True)
    for idx in range(len(file_df)):
        id = file_df.iloc[idx, 0]
        target = file_df.iloc[idx, 1]

        str_class = classes[target]
        os.makedirs(f"{data_path}/images/{str_class}", exist_ok=True)

        image = cv2.imread(f"{data_path}/train/{id}")
        cv2.imwrite(f"{data_path}/images/{str_class}/{id}", image)

if __name__ == "__main__":
    main()