import json
import numpy as np
from PIL import Image
import sys

# Load database
print("Loading character_db_small.json...")
with open('character_db_small.json', 'r') as f:
    database = json.load(f)
print(f"Database loaded! {len(database)} characters ready.")

def preprocess_image(img_path):
    """Load and preprocess image to match EMNIST format"""
    img = Image.open(img_path).convert('L')  # grayscale
    img = img.resize((28, 28))               # resize to 28x28
    
    # Convert to numpy array
    arr = np.array(img, dtype=np.float32)
    
    # Invert if background is white (EMNIST has white text on black)
    if arr.mean() > 128:
        arr = 255 - arr
    
    # Normalize to 0-1
    arr = arr / 255.0
    
    # Downsample to 14x14 (same as our compressed database)
    small = arr.reshape(14, 2, 14, 2).mean(axis=(1, 3))
    
    return small.flatten()

def knn_match(vector, k=5):
    """Find the best matching character using KNN"""
    best_char = None
    best_distance = float('inf')
    
    votes = {}
    all_distances = []

    for char, samples in database.items():
        for sample in samples:
            sample_arr = np.array(sample)
            # Euclidean distance
            dist = np.sqrt(np.sum((vector - sample_arr) ** 2))
            all_distances.append((dist, char))
    
    # Sort by distance, take top K
    all_distances.sort(key=lambda x: x[0])
    top_k = all_distances[:k]
    
    # Vote
    for dist, char in top_k:
        votes[char] = votes.get(char, 0) + 1
    
    # Winner
    best_char = max(votes, key=votes.get)
    best_distance = all_distances[0][0]
    
    # Confidence (how dominant the winner is)
    confidence = votes[best_char] / k * 100
    
    return best_char, confidence, top_k

# Get image path from command line or ask
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = input("Enter image path: ")

print(f"\nProcessing: {img_path}")

vector = preprocess_image(img_path)
char, confidence, top5 = knn_match(vector)

print(f"\nResult: '{char}'")
print(f"Confidence: {confidence:.0f}%")
print(f"\nTop 5 matches:")
for i, (dist, c) in enumerate(top5):
    print(f"  {i+1}. '{c}' — distance: {dist:.3f}")