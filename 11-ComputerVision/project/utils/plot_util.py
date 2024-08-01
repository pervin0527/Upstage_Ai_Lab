import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

def visualize_class_distribution(train_df, meta_df, save_plot=False, plot_path="class_distribution.png"):
    class_dict = dict(zip(meta_df['target'], meta_df['class_name']))
    train_df['class_name'] = train_df['target'].map(class_dict)

    class_counts = train_df['class_name'].value_counts().reset_index()
    class_counts.columns = ['class_name', 'file_count']

    plt.figure(figsize=(10, 8))
    sns.barplot(y='class_name', x='file_count', data=class_counts)
    plt.title('Number of Files per Class')
    plt.xlabel('File Count')
    plt.ylabel('Class Name')
    plt.tight_layout()

    if save_plot:
        plt.savefig(plot_path)
        print(f"Plot saved as {plot_path}")
    else:
        plt.show()

    return class_counts


def visualize_class_images(train_df, meta_df, data_path, img_size=512, save_plot=False, plot_path="class_images.png"):
    class_dict = dict(zip(meta_df['target'], meta_df['class_name']))
    train_df['class_name'] = train_df['target'].map(class_dict)
    
    unique_classes = train_df['class_name'].unique()

    for class_name in unique_classes:
        class_files = train_df[train_df['class_name'] == class_name]['ID'].head(9).tolist()

        plt.figure(figsize=(7, 7))
        for idx, file_name in enumerate(class_files):
            img_path = os.path.join(data_path, file_name)
            # img = Image.open(img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            plt.subplot(3, 3, idx + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(file_name)

        plt.suptitle(f"Class: {class_name}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        if save_plot:
            class_plot_path = plot_path.replace(".png", f"_{class_name}.png")
            plt.savefig(class_plot_path)
            print(f"Plot saved as {class_plot_path}")
        else:
            plt.show()