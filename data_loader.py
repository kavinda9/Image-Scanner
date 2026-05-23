import os
import pandas as pd
from PIL import Image
import numpy as np

BASE_PATH = r'D:\GitHub Projects\Image Scanner\data\kaggle'

IMAGE_SIZE = (128, 32)

DATASETS = [
    {'csv': os.path.join(BASE_PATH, 'written_name_train_v1.csv'),
     'img': os.path.join(BASE_PATH, 'train_v1', 'train'), 'version': 1},
    {'csv': os.path.join(BASE_PATH, 'written_name_train_v2.csv'),
     'img': os.path.join(BASE_PATH, 'train_v2', 'train'), 'version': 2},
    {'csv': os.path.join(BASE_PATH, 'written_name_validation_v1.csv'),
     'img': os.path.join(BASE_PATH, 'validation_v1', 'validation'), 'version': 1},
    {'csv': os.path.join(BASE_PATH, 'written_name_validation_v2.csv'),
     'img': os.path.join(BASE_PATH, 'validation_v2', 'validation'), 'version': 2},
    {'csv': os.path.join(BASE_PATH, 'written_name_test_v1.csv'),
     'img': os.path.join(BASE_PATH, 'test_v1', 'test'), 'version': 1},
    {'csv': os.path.join(BASE_PATH, 'written_name_test_v2.csv'),
     'img': os.path.join(BASE_PATH, 'test_v2', 'test'), 'version': 2},
]


def load_single_dataset(csv_path, img_folder, max_samples=None, version=2):
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        return [], []

    df = pd.read_csv(csv_path)
    df = df.dropna()

    if version == 1:
        filename_col = 'IMAGE'
        label_col = 'MEDICINE_NAME'
    else:
        filename_col = 'FILENAME'
        label_col = 'IDENTITY'
        df = df[df[label_col] != 'UNREADABLE']

    if max_samples:
        df = df.head(max_samples)

    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(img_folder, str(row[filename_col]))
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert('L')
            img = img.resize(IMAGE_SIZE)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(str(row[label_col]).lower().strip())
        except:
            continue

    return images, labels


def load_all_datasets(max_per_dataset=None):
    all_images = []
    all_labels = []

    for ds in DATASETS:
        print(f"Loading: {ds['csv']}")
        images, labels = load_single_dataset(
            ds['csv'], ds['img'], max_per_dataset, ds['version']
        )
        all_images.extend(images)
        all_labels.extend(labels)
        print(f"  → {len(images)} images loaded")

    print(f"\nTotal images: {len(all_images)}")
    print(f"Unique words: {len(set(all_labels))}")
    return np.array(all_images), all_labels


if __name__ == '__main__':
    images, labels = load_all_datasets(max_per_dataset=500)
    print(f"\nSample labels: {labels[:5]}")
    print(f"Image shape: {images[0].shape}")