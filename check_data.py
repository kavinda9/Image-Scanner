import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "landlord/handwriting-recognition",
  ""
)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("First 5 rows:")
print(df.head())