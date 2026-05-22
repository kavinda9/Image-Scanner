import os
import pandas as pd
from PIL import Image
import numpy as np

# Paths
BASE_PATH = r'D:\GitHub Projects\Image Scanner\data\kaggle'

TRAIN_CSV = os.path.join(BASE_PATH, 'written_name_train_v2.csv')
VAL_CSV = os.path.join(BASE_PATH, 'written_name_validation_v2.csv')
TEST_CSV = os.path.join(BASE_PATH, 'written_name_test_v2.csv')

TRAIN_IMG = os.path.join(BASE_PATH, 'train_v2', 'train')
VAL_IMG = os.path.join(BASE_PATH, 'validation_v2', 'validation')
TEST_IMG = os.path.join(BASE_PATH, 'test_v2', 'test')

IMAGE_SIZE = (128, 32)  # width x height for word images

def load_dataset(csv_path, img_folder, max_samples=None):
    df = pd.read_csv(csv_path)
    
    # Remove unreadable entries
    df = df[df['IDENTITY'] != 'UNREADABLE']
    df = df.dropna()
    
    if max_samples:
        df = df.head(max_samples)
    
    images = []
    labels = []
    
    for _, row in df.iterrows():
        img_path = os.path.join(img_folder, row['FILENAME'])
        
        if not os.path.exists(img_path):
            continue
            
        # Load and preprocess image
        img = Image.open(img_path).convert('L')  # grayscale
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0  # normalize
        
        images.append(img_array)
        labels.append(row['IDENTITY'].lower())
    
    return np.array(images), labels

if __name__ == '__main__':
    print("Loading training data...")
    train_images, train_labels = load_dataset(TRAIN_CSV, TRAIN_IMG, max_samples=1000)
    print(f"Loaded {len(train_images)} training images")
    print(f"Sample labels: {train_labels[:5]}")
    print(f"Image shape: {train_images[0].shape}")