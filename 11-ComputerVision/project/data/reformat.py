import os
import cv2
import argparse
import pandas as pd

def plant_village(data_path):
    classes = sorted(os.listdir(data_path))

    class_dict = {}
    for idx, label in enumerate(classes):
        class_dict[idx] = label

    meta_data = [{'target': idx, 'class_name': label} for idx, label in class_dict.items()]
    meta_df = pd.DataFrame(meta_data)
    meta_df.to_csv("../datasets/meta.csv", index=False)

    os.makedirs("../datasets/train", exist_ok=True)
    total_data_list = []
    for class_idx, label in enumerate(classes):
        files = os.listdir(f"{data_path}/{label}")
        for file in files:
            if file != "svn-r6Yb5c":

                total_data_list.append({'ID' : file, 'target' : class_idx})
                image = cv2.imread(f"{data_path}/{label}/{file}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"../datasets/train/{file}", image)
            
    train_df = pd.DataFrame(total_data_list)
    train_df.to_csv("../datasets/train.csv", index=False)


def plant_pathology(data_path):
    df = pd.read_csv(f"{data_path}/train.csv")
    classes = list(df['labels'].unique())
    
    class_dict = dict()
    for idx, label in enumerate(classes):
        class_dict[idx] = label
        
    meta_data = [{'target': idx, 'class_name': label} for idx, label in class_dict.items()]
    meta_df = pd.DataFrame(meta_data)
    meta_df.to_csv(f"{data_path}/meta.csv", index=False)

    df = df.rename(columns={'image': 'ID'})
    reverse_class_dict = {v: k for k, v in class_dict.items()}
    df['labels'] = df['labels'].map(reverse_class_dict)
    df = df.rename(columns={'labels': 'target'})

    df.to_csv(f"{data_path}/train.csv", index=False)

    # os.rename(f"{data_path}/train_images", f"{data_path}/train")
    # os.rename(f"{data_path}/test_images", f"{data_path}/test")


def main(args):
    if args.data_name == "plant_village":
        plant_village(args.data_path)
    elif args.data_name == "plant_pathology":
        plant_pathology(args.data_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process config path.")
    parser.add_argument('--data_name', type=str, help='Path to the config file')
    parser.add_argument('--data_path', type=str, help='Path to the config file')
    args = parser.parse_args()
    
    main(args)