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

        map_dict = {"45f0d2dfc7e47c03.jpg" : 7,
                    "aec62dced7af97cd.jpg" : 14,
                    "0583254a73b48ece.jpg" : 10,
                    "1ec14a14bbe633db.jpg" : 7,
                    "c5182ab809478f12.jpg" : 14,
                    "8646f2c3280a4f49.jpg" : 3}
        if id in map_dict:
            target = map_dict[id]

        str_class = classes[target]
        os.makedirs(f"{data_path}/images/{str_class}", exist_ok=True)

        image = cv2.imread(f"{data_path}/train/{id}")
        cv2.imwrite(f"{data_path}/images/{str_class}/{id}", image)

if __name__ == "__main__":
    main()