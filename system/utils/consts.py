"""Module for defining application-level constants."""
import string

IMAGE_SIZE = (64, 64)
CLASSES = [c for c in string.ascii_lowercase + string.digits]

# Model names
EMNIST_MODEL = 'emnist.h5'
KAGGLE_MODEL = 'kaggle.h5'

DOCX_MIME_TYPE = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
CHARACTER_PADDING_RATIO = 0.25
INFINITY = 2**64
EVALUATION_RESULTS_DIR = './evaluation_results'
