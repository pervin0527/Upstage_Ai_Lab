import os
import pandas as pd

def make_save_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_dataset(path, file_name, dataset):
    make_save_dir(path)

    df = pd.DataFrame(dataset)
    df.to_csv(f"{path}/{file_name}.csv", encoding='utf-8-sig')

    print(f"Data saved at {path}/{file_name}.csv")