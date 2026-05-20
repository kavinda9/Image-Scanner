import numpy as np
import json
import os
from torchvision import datasets, transforms

print("Loading EMNIST dataset...")

transform = transforms.ToTensor()

train_data = datasets.EMNIST(
    root='./data',
    split='byclass',
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.EMNIST(
    root='./data',
    split='byclass',
    train=False,
    download=True,
    transform=transform
)

print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

def label_to_char(label):
    if label <= 9:
        return str(label)
    elif label <= 35:
        return chr(label - 10 + ord('A'))
    else:
        return chr(label - 36 + ord('a'))

SAMPLES_PER_CLASS = 50
NUM_CLASSES = 62

# Collect per class directly — no need to load everything at once
print("\nCollecting samples per character (memory efficient)...")

database = {label_to_char(i): [] for i in range(NUM_CLASSES)}
counts = {label_to_char(i): 0 for i in range(NUM_CLASSES)}
needed = SAMPLES_PER_CLASS

# Go through train data first
print("Scanning training data...")
for img, label in train_data:
    char = label_to_char(label)
    if counts[char] < needed:
        vector = img.numpy().flatten().tolist()
        database[char].append(vector)
        counts[char] += 1

# Fill remaining from test data if needed
print("Scanning test data for any missing...")
for img, label in test_data:
    char = label_to_char(label)
    if counts[char] < needed:
        vector = img.numpy().flatten().tolist()
        database[char].append(vector)
        counts[char] += 1

# Check all classes collected
print("\nSummary:")
for i in range(NUM_CLASSES):
    char = label_to_char(i)
    print(f"  '{char}' → {counts[char]} samples")

print("\nSaving to character_db.json ...")
with open('character_db.json', 'w') as f:
    json.dump(database, f)

size_kb = os.path.getsize('character_db.json') / 1024
print(f"\nDone! character_db.json created ({size_kb:.1f} KB)")
print(f"Total characters: {len(database)}")