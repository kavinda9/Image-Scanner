import json
import numpy as np
import os

print("Loading character_db.json...")
with open('character_db.json', 'r') as f:
    database = json.load(f)

print("Compressing vectors...")

compressed = {}
for char, samples in database.items():
    compressed_samples = []
    for vector in samples:
        arr = np.array(vector)
        # Reshape to 28x28
        img = arr.reshape(28, 28)
        # Downsample to 14x14 (quarter the data)
        small = img.reshape(14, 2, 14, 2).mean(axis=(1, 3))
        # Round to 2 decimal places
        rounded = np.round(small.flatten(), 2).tolist()
        compressed_samples.append(rounded)
    compressed[char] = compressed_samples
    print(f"  '{char}' compressed")

print("\nSaving to character_db_small.json...")
with open('character_db_small.json', 'w') as f:
    json.dump(compressed, f, separators=(',', ':'))

original_kb = os.path.getsize('character_db.json') / 1024
compressed_kb = os.path.getsize('character_db_small.json') / 1024
print(f"\nOriginal:   {original_kb:.1f} KB")
print(f"Compressed: {compressed_kb:.1f} KB")
print(f"Reduced by: {100 - (compressed_kb/original_kb*100):.1f}%")