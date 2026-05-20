import numpy as np
import json
import os
from emnist import extract_training_samples, extract_test_samples
from PIL import Image

print("Loading EMNIST dataset... (first time may take a while to download)")

# Load EMNIST balanced dataset (has a-z, A-Z, 0-9)
x_train, y_train = extract_training_samples('byclass')
x_test, y_test = extract_test_samples('byclass')

# Combine train and test
x_all = np.concatenate([x_train, x_test])
y_all = np.concatenate([y_train, y_test])

print(f"Total images loaded: {len(x_all)}")

# EMNIST byclass labels:
# 0-9   → digits '0'-'9'
# 10-35 → uppercase 'A'-'Z'
# 36-61 → lowercase 'a'-'z'

def label_to_char(label):
    if label <= 9:
        return str(label)
    elif label <= 35:
        return chr(label - 10 + ord('A'))
    else:
        return chr(label - 36 + ord('a'))

SAMPLES_PER_CLASS = 50
NUM_CLASSES = 62

print(f"Picking {SAMPLES_PER_CLASS} samples per character...")

database = {}

for class_id in range(NUM_CLASSES):
    char = label_to_char(class_id)
    
    # Get all indices for this class
    indices = np.where(y_all == class_id)[0]
    
    # Pick 50 random samples
    chosen = np.random.choice(indices, SAMPLES_PER_CLASS, replace=False)
    
    samples = []
    for idx in chosen:
        img = x_all[idx]  # 28x28 grayscale image
        
        # Flatten to 1D array and normalize to 0-1
        vector = (img.flatten() / 255.0).tolist()
        samples.append(vector)
    
    database[char] = samples
    print(f"  '{char}' → {SAMPLES_PER_CLASS} samples collected")

print("\nSaving to character_db.json ...")

with open('character_db.json', 'w') as f:
    json.dump(database, f)

size_kb = os.path.getsize('character_db.json') / 1024
print(f"\nDone! character_db.json created ({size_kb:.1f} KB)")
print(f"Total characters: {len(database)}")
print(f"Total samples: {len(database) * SAMPLES_PER_CLASS}")