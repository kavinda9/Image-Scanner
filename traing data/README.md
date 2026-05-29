Input Image
│
▼
segment.py ← Detect & crop word/line regions (OpenCV + EAST)
│
▼
data_loader.py ← Preprocess, normalise, augment
│
├──► model.py (Stage 1) ← CNN character classifier (trained on EMNIST)
│
└──► model.py (Stage 2) ← CRNN + CTC word recogniser (trained on Kaggle)
│
▼
inference.py ← CTC decode → spell-check → output text

```

## Folder Structure

```

prescription_ocr/
├── data/
│ ├── emnist/ # Raw EMNIST IDX or .npz files
│ ├── kaggle_prescriptions/ # Kaggle word images + labels.csv
│ └── processed/ # Output of prepare_database.py
├── models/ # Saved .h5 model weights
├── logs/ # TensorBoard logs
├── outputs/ # Evaluation plots, sample grids
├── prepare_database.py # Data ingestion & preprocessing
├── compress_db.py # Pack splits into HDF5
├── check_data.py # Sanity checks & visualisations
├── segment.py # Text detection / word segmentation
├── data_loader.py # tf.data pipelines
├── model.py # CNN (char) + CRNN (word) architectures
├── train.py # Training orchestration
├── evaluate.py # CER / WER evaluation
├── inference.py # End-to-end inference + REST API
├── utils.py # Augmentation, metrics, CTC helpers
├── character_db.json # Class-index ↔ character mappings
└── requirements.txt

````

## Quick Start

### 1. Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
````

### 2. Prepare data

```bash
# Place EMNIST files in data/emnist/
# Place Kaggle images + labels.csv in data/kaggle_prescriptions/

python prepare_database.py
python compress_db.py
python check_data.py --visualise
```

### 3. Train

```bash
# Stage 1 – character model
python train.py --stage char --epochs 30

# Stage 2 – word CRNN
python train.py --stage word --epochs 50 --pretrained models/char_model.h5
```

### 4. Evaluate

```bash
python evaluate.py --model models/word_model.h5 --split test
```

### 5. Inference

```bash
# Single image
python inference.py --image path/to/prescription.jpg

# Start REST API
uvicorn inference:app --host 0.0.0.0 --port 8000
```

## Training Strategy

| Stage | Dataset | Model      | Loss         | Key Metric            |
| ----- | ------- | ---------- | ------------ | --------------------- |
| 1     | EMNIST  | CNN        | CrossEntropy | Char accuracy         |
| 2     | Kaggle  | CRNN + CTC | CTC          | CER / WER             |
| 3     | Mixed   | Fine-tune  | CTC          | Prescription accuracy |

## Safety Notes

- Any word with confidence < threshold is **flagged for human review**
- Recognised drug names are fuzzy-matched against a medical dictionary
- All inference results are logged for audit purposes

## Dependencies

See `requirements.txt`. Key libraries:

- TensorFlow ≥ 2.13
- OpenCV, Pillow, NumPy, SciPy
- pandas, scikit-learn, seaborn, matplotlib
- FastAPI + Uvicorn (inference API)
- h5py (HDF5 storage)
